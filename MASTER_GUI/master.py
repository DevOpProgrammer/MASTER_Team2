import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
# from torch.nn.modules.normalization import RMSNorm  # Not available in PyTorch 1.11.0
from torchinfo import summary

import math
import yaml

from base_model import SequenceModel

# Custom implementation of RMSNorm since it's not in PyTorch 1.11.0
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm



# Read all config from config.yaml
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except Exception as e:
        print(f"Warning: Could not read config.yaml, using empty config. Error: {e}")
        return {}

CONFIG = load_config()

def get_config(key, default=None):
    return CONFIG.get(key, default)

def get_norm_layer(dim):
    """Returns the appropriate normalization layer based on the config setting."""
    norm_type = get_config(key="NORM_TYPE", default="layer_norm")
    # print(f"Using normalization type: {norm_type}")
    if norm_type == "layer_norm":
        return LayerNorm(dim, eps=1e-5)
    else:  # "rms_norm" by default
        return RMSNorm(dim)

def get_layer_norm_type():
    layer_norm_type = get_config(key="LAYER_NORM_TYPE", default="peri_norm")
    # print(f"Using layer normalization type: {layer_norm_type}")
    if layer_norm_type == "pre_norm":
        layer_norm_type_arr = [1, 0, 1, 0]
    elif layer_norm_type == "peri_norm":
        layer_norm_type_arr = [1, 1, 1, 1]
    elif layer_norm_type == "post_norm":
        layer_norm_type_arr = [0, 1, 0, 1]
    else:
        raise ValueError(f"Unknown layer normalization type: {layer_norm_type}")
    return layer_norm_type_arr

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = get_norm_layer(d_model)    # self.norm1 = LayerNorm(d_model, eps=1e-5)

        # Norm after attention
        self.norm1_2 = get_norm_layer(d_model)

        # FFN layerNorm
        self.norm2 = get_norm_layer(d_model)    # self.norm2 = LayerNorm(d_model, eps=1e-5)

        # Norm after FFN
        self.norm2_2 = get_norm_layer(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        layer_norm_type_arr = get_layer_norm_type()
        # Apply first normalization
        if layer_norm_type_arr[0] == 1:
            x = self.norm1(x)
        
        # Apply attention operation
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)
        
        # Apply second normalization (always in both modes - peri_norm uses this one)
        if layer_norm_type_arr[1] == 1:
            att_output = self.norm1_2(att_output)

        # Residual connection
        xt = x + att_output
        
        # Apply third normalization
        if layer_norm_type_arr[2] == 1:
            xt = self.norm2(xt)
        
        # Apply FFN
        ffn_output = self.ffn(xt)

        # Apply fourth normalization only in peri_norm mode
        if layer_norm_type_arr[3] == 1:
            ffn_output = self.norm2_2(ffn_output)
            
        # Final residual connection
        att_output = xt + ffn_output

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = get_norm_layer(d_model)    # self.norm1 = LayerNorm(d_model, eps=1e-5)

        # Norm after attention
        self.norm1_2 = get_norm_layer(d_model)

        # FFN layerNorm
        self.norm2 = get_norm_layer(d_model)    # self.norm2 = LayerNorm(d_model, eps=1e-5)

        # Norm after FFN
        self.norm2_2 = get_norm_layer(d_model)

        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        layer_norm_type_arr = get_layer_norm_type()
        # Apply first normalization
        if layer_norm_type_arr[0] == 1:
            x = self.norm1(x)
        
        # Apply attention operation
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)
        
        # Apply second normalization (always in both modes - peri_norm uses this one)
        if layer_norm_type_arr[1] == 1:
            att_output = self.norm1_2(att_output)

        # Residual connection
        xt = x + att_output
        
        # Apply third normalization (always in both modes)
        if layer_norm_type_arr[2] == 1:
            xt = self.norm2(xt)

        # Apply FFN
        ffn_output = self.ffn(xt)

        # Apply fourth normalization only in peri_norm mode
        if layer_norm_type_arr[3] == 1:
            ffn_output = self.norm2_2(ffn_output)
            
        # Final residual connection
        att_output = xt + ffn_output

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
       
        output = self.layers(src).squeeze(-1)

        return output


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()

if __name__ == "__main__":
    # Example usage
    model = MASTERModel(
        d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
        gate_input_start_index=158, gate_input_end_index=221,
        T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5,
        n_epochs=40, lr=1e-5, GPU=0, seed=42, train_stop_loss_thred=0.95,
        save_path='best_param/', save_prefix='csi300_opensource'
    )
    # Set up correct batch_size and datasize according to main.py
    batch_size = 32
    seq_len = 60
    d_feat = 158 + (221 - 158)  # total input features (src + gate_input)
    # But MASTERModel expects input of shape (batch_size, seq_len, total_features)
    # The model uses only the first 158 as src, and 158:221 as gate_input

    # Create dummy input and move it to the same device as the model
    dummy_input = torch.randn(batch_size, seq_len, 221).to(model.device)
    
    # Ensure model is on the correct device
    model.model = model.model.to(model.device)
    
    # Print device information for debugging
    print(f"Model device: {next(model.model.parameters()).device}")
    print(f"Input device: {dummy_input.device}")
    
    # Run model summary
    summary(model.model, input_data=dummy_input)