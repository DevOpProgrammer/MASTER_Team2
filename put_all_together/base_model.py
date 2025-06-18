import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
import pickle
import os



def ranknet_loss(pred, label):
    """
    Pairwise RankNet Loss for ranking consistency.
    pred, label: [N] where N is number of stocks in the day
    """
    diff_label = label.unsqueeze(1) - label.unsqueeze(0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    
    sign = torch.sign(diff_label)  # ground truth order
    mask = (sign != 0)  # ignore equal values

    rank_loss = torch.log1p(torch.exp(-diff_pred * sign))
    rank_loss = rank_loss[mask]
    return torch.mean(rank_loss)

def loss_with_ranknet(pred, label, alpha=0.1):
    """
    Combines MSE loss with RankNet loss using weight alpha.
    """
    mask = ~torch.isnan(label)
    pred, label = pred[mask], label[mask]

    mse = (pred - label) ** 2
    mse_loss = torch.mean(mse)
    rank_loss = ranknet_loss(pred, label)

    return mse_loss + alpha * rank_loss



def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]



class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)
    

    def loss_fn_with_ranknet(self, pred, label, alpha=0.1):
        return loss_with_ranknet(pred, label, alpha=alpha)



    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 0
    def _save_training_history(self, train_loss, ic, icir, ric, ricir, filename=None, dir='./training_history', seed=0, epc_num=0):
        """
        Save training losses and evaluation metrics to a pickle file.

        Args:
            ic (list): List of IC values per epoch or run.
            icir (list): List of ICIR values per epoch or run.
            ric (list): List of RIC values per epoch or run.
            ricir (list): List of RICIR values per epoch or run.
            filename (str): Path to save the pickle file.
        """
        history = {
            'train_loss': train_loss,
            'ic': ic,
            'icir': icir,
            'ric': ric,
            'ricir': ricir
        }

        file_path = os.path.join(dir, f"{filename}-sead-{seed}-epc-{epc_num}.pkl")
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)

    def fit(self, dl_train, dl_valid, seed):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        train_hist_filename = 'training_history'
        train_losses = []
        ICs = []
        ICIRs = []
        RICs = []
        RICIRs = []
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
                ICs.append(metrics['IC'])
                ICIRs.append(metrics['ICIR'])
                RICs.append(metrics['RIC'])
                RICIRs.append(metrics['RICIR'])

                if step > 40:
                    self._save_training_history(
                        train_loss=train_losses,
                        ic=ICs, icir=ICIRs,
                        ric=RICs, ricir=RICIRs,
                        filename=train_hist_filename,
                        seed=seed,
                        epc_num=step
                    )
                    best_param = self.model.state_dict()
                    dir = f"/home/l/下載/MASTER/{seed}/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    file_path = os.path.join(dir, f"{step}_csi300_original_{seed}")
                    torch.save(best_param, f"{file_path}.pkl")
            else: 
                print("Epoch %d, train_loss %.6f" % (step, train_loss))
        


    def predict(self, dl_test):
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            
            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return predictions, metrics