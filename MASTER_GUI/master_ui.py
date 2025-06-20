import sys
import os
import pickle
import numpy as np
import time
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QLineEdit, 
                            QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox, 
                            QTextEdit, QProgressBar, QFileDialog, QCheckBox,
                            QGridLayout, QTabWidget, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

# Import the MASTERModel for training
from master import MASTERModel

class TrainingWorker(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            # Load data
            self.update_signal.emit("Loading data...")
            
            # Get file paths from config
            train_data_path = self.config.get('train_data_path')
            valid_data_path = self.config.get('valid_data_path')
            test_data_path = self.config.get('test_data_path')
            
            if not train_data_path or not valid_data_path or not test_data_path:
                self.update_signal.emit("Error: Data files not specified")
                return
            
            try:
                with open(train_data_path, 'rb') as f:
                    dl_train = pickle.load(f)
                self.update_signal.emit(f"Loaded training data from {train_data_path}")
                
                with open(valid_data_path, 'rb') as f:
                    dl_valid = pickle.load(f)
                self.update_signal.emit(f"Loaded validation data from {valid_data_path}")
                
                with open(test_data_path, 'rb') as f:
                    dl_test = pickle.load(f)
                self.update_signal.emit(f"Loaded test data from {test_data_path}")
                
                self.update_signal.emit("Data loaded successfully.")
            except Exception as e:
                self.update_signal.emit(f"Error loading data: {str(e)}")
                return
            
            # Setup training parameters
            train_hist_file = time.strftime("%Y%m%d_%H%M%S")
            
            # Parse parameters from config
            d_feat = self.config.get('d_feat', 158)
            d_model = self.config.get('d_model', 256)
            t_nhead = self.config.get('t_nhead', 4)
            s_nhead = self.config.get('s_nhead', 2)
            dropout = self.config.get('dropout', 0.5)
            gate_input_start_index = self.config.get('gate_input_start_index', 158)
            gate_input_end_index = self.config.get('gate_input_end_index', 221)
            
            # Get universe from config
            universe = self.config.get('universe', 'csi300')
            
            # Set beta based on universe
            if universe == 'csi300':
                beta = self.config.get('beta', 5)
            elif universe == 'csi800':
                beta = self.config.get('beta', 2)
            else:
                beta = self.config.get('beta', 5)
            
            n_epoch = self.config.get('n_epoch', 40)
            lr = self.config.get('lr', 1e-5)
            GPU = self.config.get('GPU', 0)
            train_stop_loss_thred = self.config.get('train_stop_loss_thred', 0.95)
            seeds = self.config.get('seeds', [0])
            
            prefix = self.config.get('prefix', 'opensource')
            
            # Create necessary directories if they don't exist
            os.makedirs('best_param', exist_ok=True)
            os.makedirs('training_history', exist_ok=True)
            
            ic = []
            icir = []
            ric = []
            ricir = []
            
            # Train and evaluate for each seed
            total_seeds = len(seeds)
            for i, seed in enumerate(seeds):
                self.update_signal.emit(f"Training model with seed {seed} ({i+1}/{total_seeds})...")
                
                # Initialize model
                model = MASTERModel(
                    d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
                    T_dropout_rate=dropout, S_dropout_rate=dropout,
                    beta=beta, gate_input_end_index=gate_input_end_index, 
                    gate_input_start_index=gate_input_start_index,
                    n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, 
                    train_stop_loss_thred=train_stop_loss_thred,
                    save_path=f'best_param/{train_hist_file}_{seed}.pkl', 
                    save_prefix=f'{universe}_{prefix}'
                )
                
                # Train model
                start = time.time()
                model.fit(dl_train, dl_valid, train_hist_filename=f'{train_hist_file}_{seed}')
                
                # Test model
                self.update_signal.emit(f"Testing model with seed {seed}...")
                predictions, metrics = model.predict(dl_test)
                
                running_time = time.time() - start
                
                self.update_signal.emit(f'Seed: {seed} time cost: {running_time:.2f} sec')
                self.update_signal.emit(f'Metrics for seed {seed}: {metrics}')
                
                # Store metrics
                ic.append(metrics['IC'])
                icir.append(metrics['ICIR'])
                ric.append(metrics['RIC'])
                ricir.append(metrics['RICIR'])
                
                # Update progress
                progress = int((i + 1) / total_seeds * 100)
                self.progress_signal.emit(progress)
            
            # Calculate average metrics
            results = {
                'IC_mean': np.mean(ic),
                'IC_std': np.std(ic),
                'ICIR_mean': np.mean(icir),
                'ICIR_std': np.std(icir),
                'RIC_mean': np.mean(ric),
                'RIC_std': np.std(ric),
                'RICIR_mean': np.mean(ricir),
                'RICIR_std': np.std(ricir),
                'train_hist_file': train_hist_file,
                'seeds': seeds
            }
            
            # Append training history statistics
            for seed in seeds:
                hist_file = f'training_history/{train_hist_file}_{seed}.pkl'
                if os.path.exists(hist_file):
                    with open(hist_file, 'rb') as f:
                        history = pickle.load(f)
                    history['tested_ic_mean'] = np.mean(ic)
                    history['tested_ic_std'] = np.std(ic)
                    history['tested_icir_mean'] = np.mean(icir)
                    history['tested_icir_std'] = np.std(icir)
                    history['tested_ric_mean'] = np.mean(ric)
                    history['tested_ric_std'] = np.std(ric)
                    history['tested_ricir_mean'] = np.mean(ricir)
                    history['tested_ricir_std'] = np.std(ricir)
                    with open(hist_file, 'wb') as f:
                        pickle.dump(history, f)
            
            self.update_signal.emit("Training complete!")
            self.finished_signal.emit(results)
            
        except Exception as e:
            self.update_signal.emit(f"Error during training: {str(e)}")


class EvaluationWorker(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            # Get parameters from config
            param_path = self.config.get('model_path')
            test_data_path = self.config.get('test_data_path')
            universe = self.config.get('universe', 'csi300')
            
            # Model parameters
            d_feat = self.config.get('d_feat', 158)
            d_model = self.config.get('d_model', 256)
            t_nhead = self.config.get('t_nhead', 4)
            s_nhead = self.config.get('s_nhead', 2)
            dropout = self.config.get('dropout', 0.5)
            gate_input_start_index = self.config.get('gate_input_start_index', 158)
            gate_input_end_index = self.config.get('gate_input_end_index', 221)
            
            # Other parameters
            n_epoch = self.config.get('n_epoch', 40)
            lr = self.config.get('lr', 1e-5)
            GPU = self.config.get('GPU', 0)
            train_stop_loss_thred = self.config.get('train_stop_loss_thred', 0.95)
            seeds = self.config.get('seeds', [0])
            
            self.update_signal.emit(f"Loading test data from {test_data_path}...")
            try:
                # Load test data
                with open(test_data_path, 'rb') as f:
                    dl_test = pickle.load(f)
                self.update_signal.emit(f"Test data loaded successfully from {test_data_path}")
            except Exception as e:
                self.update_signal.emit(f"Error loading test data: {str(e)}")
                return
            
            # Initialize metrics aggregation
            ic_values = []
            icir_values = []
            ric_values = []
            ricir_values = []
            
            total_seeds = len(seeds)
            for i, seed in enumerate(seeds):
                self.update_signal.emit(f"Evaluating model with seed {seed} ({i+1}/{total_seeds})...")
                
                # Initialize model with current seed
                self.update_signal.emit(f"Initializing model with seed {seed}...")
                model = MASTERModel(
                    d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, 
                    T_dropout_rate=dropout, S_dropout_rate=dropout,
                    beta=self.config.get('beta', 5), 
                    gate_input_end_index=gate_input_end_index, 
                    gate_input_start_index=gate_input_start_index,
                    n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, 
                    train_stop_loss_thred=train_stop_loss_thred,
                    save_path='model/', save_prefix=universe
                )
                
                # Load model parameters
                self.update_signal.emit(f"Loading model parameters from {param_path}...")
                try:
                    model.load_param(param_path)
                    self.update_signal.emit("Model parameters loaded successfully")
                except Exception as e:
                    self.update_signal.emit(f"Error loading model parameters: {str(e)}")
                    continue
                
                # Make predictions
                self.update_signal.emit(f"Evaluating model on test data with seed {seed}...")
                predictions, metrics = model.predict(dl_test)
                
                self.update_signal.emit(f"Seed {seed} evaluation complete")
                self.update_signal.emit(f"Metrics for seed {seed}: {metrics}")
                
                # Store metrics for this seed
                ic_values.append(metrics['IC'])
                icir_values.append(metrics['ICIR'])
                ric_values.append(metrics['RIC'])
                ricir_values.append(metrics['RICIR'])
            
            # Calculate average metrics across all seeds
            final_metrics = {
                'IC': float(np.mean(ic_values)),
                'IC_std': float(np.std(ic_values)),
                'ICIR': float(np.mean(icir_values)),
                'ICIR_std': float(np.std(icir_values)),
                'RIC': float(np.mean(ric_values)),
                'RIC_std': float(np.std(ric_values)),
                'RICIR': float(np.mean(ricir_values)),
                'RICIR_std': float(np.std(ricir_values)),
                'seeds_used': seeds
            }
            
            self.update_signal.emit("Evaluation complete across all seeds!")
            self.update_signal.emit(f"Average metrics: {final_metrics}")
            
            # Return metrics
            self.finished_signal.emit(final_metrics)
            
        except Exception as e:
            self.update_signal.emit(f"Error during evaluation: {str(e)}")


class MASTERModelUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MASTER Model Training UI")
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create training tab
        self.training_tab = QWidget()
        self.tabs.addTab(self.training_tab, "Training")
        
        # Create evaluation tab
        self.evaluation_tab = QWidget()
        self.tabs.addTab(self.evaluation_tab, "Evaluation")
        
        # Setup training tab UI
        self.setup_training_tab()
        
        # Setup evaluation tab UI
        self.setup_evaluation_tab()
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Auto-fill data paths on startup (after UI is fully setup)
        QApplication.processEvents()
        self.auto_fill_data_paths()
        self.update_eval_data_path()  # Auto-fill evaluation test data path
    
    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        
        # Data configuration group
        data_group = QGroupBox("Data Configuration")
        data_layout = QGridLayout(data_group)
        
        # Universe selection
        data_layout.addWidget(QLabel("Universe:"), 0, 0)
        self.universe_combo = QComboBox()
        self.universe_combo.addItems(["csi300", "csi800"])
        self.universe_combo.currentIndexChanged.connect(self.auto_fill_data_paths)
        data_layout.addWidget(self.universe_combo, 0, 1)
        
        # Prefix selection
        data_layout.addWidget(QLabel("Prefix:"), 1, 0)
        self.prefix_combo = QComboBox()
        self.prefix_combo.addItems(["opensource", "original"])
        self.prefix_combo.currentIndexChanged.connect(self.auto_fill_data_paths)
        data_layout.addWidget(self.prefix_combo, 1, 1)
        
        # Training data file
        data_layout.addWidget(QLabel("Training Data:"), 2, 0)
        train_data_layout = QHBoxLayout()
        self.train_data_path = QLineEdit()
        self.train_data_path.setPlaceholderText("Select training data file (.pkl)")
        train_data_layout.addWidget(self.train_data_path)
        self.train_data_browse = QPushButton("Browse")
        self.train_data_browse.clicked.connect(self.browse_train_data)
        train_data_layout.addWidget(self.train_data_browse)
        data_layout.addLayout(train_data_layout, 2, 1)
        
        # Validation data file
        data_layout.addWidget(QLabel("Validation Data:"), 3, 0)
        valid_data_layout = QHBoxLayout()
        self.valid_data_path = QLineEdit()
        self.valid_data_path.setPlaceholderText("Select validation data file (.pkl)")
        valid_data_layout.addWidget(self.valid_data_path)
        self.valid_data_browse = QPushButton("Browse")
        self.valid_data_browse.clicked.connect(self.browse_valid_data)
        valid_data_layout.addWidget(self.valid_data_browse)
        data_layout.addLayout(valid_data_layout, 3, 1)
        
        # Test data file
        data_layout.addWidget(QLabel("Test Data:"), 4, 0)
        test_data_layout = QHBoxLayout()
        self.test_data_path = QLineEdit()
        self.test_data_path.setPlaceholderText("Select test data file (.pkl)")
        test_data_layout.addWidget(self.test_data_path)
        self.test_data_browse = QPushButton("Browse")
        self.test_data_browse.clicked.connect(self.browse_test_data)
        test_data_layout.addWidget(self.test_data_browse)
        data_layout.addLayout(test_data_layout, 4, 1)
        
        # No Quick Load button needed since we auto-fill on startup
        
        layout.addWidget(data_group)
        
        # Model parameters group
        model_group = QGroupBox("Model Parameters")
        model_layout = QGridLayout(model_group)
        
        # Feature dimension
        model_layout.addWidget(QLabel("Feature Dimension (d_feat):"), 0, 0)
        self.d_feat_spin = QSpinBox()
        self.d_feat_spin.setRange(1, 1000)
        self.d_feat_spin.setValue(158)
        model_layout.addWidget(self.d_feat_spin, 0, 1)
        
        # Model dimension
        model_layout.addWidget(QLabel("Model Dimension (d_model):"), 1, 0)
        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(16, 1024)
        self.d_model_spin.setValue(256)
        model_layout.addWidget(self.d_model_spin, 1, 1)
        
        # Temporal attention heads
        model_layout.addWidget(QLabel("Temporal Attention Heads (t_nhead):"), 2, 0)
        self.t_nhead_spin = QSpinBox()
        self.t_nhead_spin.setRange(1, 16)
        self.t_nhead_spin.setValue(4)
        model_layout.addWidget(self.t_nhead_spin, 2, 1)
        
        # Spatial attention heads
        model_layout.addWidget(QLabel("Spatial Attention Heads (s_nhead):"), 3, 0)
        self.s_nhead_spin = QSpinBox()
        self.s_nhead_spin.setRange(1, 16)
        self.s_nhead_spin.setValue(2)
        model_layout.addWidget(self.s_nhead_spin, 3, 1)
        
        # Dropout rate
        model_layout.addWidget(QLabel("Dropout Rate:"), 4, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.5)
        model_layout.addWidget(self.dropout_spin, 4, 1)
        
        # Gate input indices
        model_layout.addWidget(QLabel("Gate Input Start Index:"), 5, 0)
        self.gate_start_spin = QSpinBox()
        self.gate_start_spin.setRange(0, 1000)
        self.gate_start_spin.setValue(158)
        model_layout.addWidget(self.gate_start_spin, 5, 1)
        
        model_layout.addWidget(QLabel("Gate Input End Index:"), 6, 0)
        self.gate_end_spin = QSpinBox()
        self.gate_end_spin.setRange(0, 1000)
        self.gate_end_spin.setValue(221)
        model_layout.addWidget(self.gate_end_spin, 6, 1)
        
        # Beta
        model_layout.addWidget(QLabel("Beta:"), 7, 0)
        self.beta_spin = QSpinBox()
        self.beta_spin.setRange(1, 10)
        self.beta_spin.setValue(5)
        model_layout.addWidget(self.beta_spin, 7, 1)
        
        layout.addWidget(model_group)
        
        # Training parameters group
        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout(train_group)
        
        # Number of epochs
        train_layout.addWidget(QLabel("Number of Epochs:"), 0, 0)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 200)
        self.epoch_spin.setValue(40)
        train_layout.addWidget(self.epoch_spin, 0, 1)
        
        # Learning rate
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1e-2)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setSingleStep(1e-6)
        self.lr_spin.setValue(1e-5)
        train_layout.addWidget(self.lr_spin, 1, 1)
        
        # GPU ID
        train_layout.addWidget(QLabel("GPU ID:"), 2, 0)
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 8)
        self.gpu_spin.setValue(0)
        train_layout.addWidget(self.gpu_spin, 2, 1)
        
        # Train stop loss threshold
        train_layout.addWidget(QLabel("Train Stop Loss Threshold:"), 3, 0)
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 0.99)
        self.stop_loss_spin.setSingleStep(0.01)
        self.stop_loss_spin.setValue(0.95)
        train_layout.addWidget(self.stop_loss_spin, 3, 1)
        
        # Seeds
        train_layout.addWidget(QLabel("Seeds:"), 4, 0)
        self.seeds_edit = QLineEdit("0,1,2,3,4")
        train_layout.addWidget(self.seeds_edit, 4, 1)
        
        layout.addWidget(train_group)
        
        # Training control
        control_layout = QHBoxLayout()
        
        # Start training button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_button)
        
        # Stop training button
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
    
    def setup_evaluation_tab(self):
        layout = QVBoxLayout(self.evaluation_tab)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout(model_group)
        
        # Model path
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_edit = QLineEdit()
        model_layout.addWidget(self.model_path_edit, 0, 1)
        
        # Browse button
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_button, 0, 2)
        
        layout.addWidget(model_group)
        
        # Data configuration group
        data_group = QGroupBox("Data Configuration")
        data_layout = QGridLayout(data_group)
        
        # Universe selection for evaluation
        # data_layout.addWidget(QLabel("Universe:"), 0, 0)
        # self.eval_universe_combo = QComboBox()
        # self.eval_universe_combo.addItems(["csi300", "csi800"])
        # self.eval_universe_combo.currentIndexChanged.connect(self.update_eval_data_path)
        # data_layout.addWidget(self.eval_universe_combo, 0, 1)
        
        # Prefix selection
        # data_layout.addWidget(QLabel("Prefix:"), 1, 0)
        # self.eval_prefix_combo = QComboBox()
        # self.eval_prefix_combo.addItems(["opensource", "original"])
        # self.eval_prefix_combo.currentIndexChanged.connect(self.update_eval_data_path)
        # data_layout.addWidget(self.eval_prefix_combo, 1, 1)
        
        # Test data file
        data_layout.addWidget(QLabel("Test Data:"), 2, 0)
        eval_test_data_layout = QHBoxLayout()
        self.eval_test_data_path = QLineEdit()
        self.eval_test_data_path.setPlaceholderText("Select test data file (.pkl)")
        eval_test_data_layout.addWidget(self.eval_test_data_path)
        self.eval_test_data_browse = QPushButton("Browse")
        self.eval_test_data_browse.clicked.connect(self.browse_eval_test_data)
        eval_test_data_layout.addWidget(self.eval_test_data_browse)
        data_layout.addLayout(eval_test_data_layout, 2, 1)
        
        layout.addWidget(data_group)
        
        # Evaluation parameters
        eval_group = QGroupBox("Evaluation Parameters")
        eval_layout = QGridLayout(eval_group)
        
        # GPU ID
        eval_layout.addWidget(QLabel("GPU ID:"), 0, 0)
        self.eval_gpu_spin = QSpinBox()
        self.eval_gpu_spin.setRange(0, 8)
        self.eval_gpu_spin.setValue(0)
        eval_layout.addWidget(self.eval_gpu_spin, 0, 1)
        
        # Seeds
        eval_layout.addWidget(QLabel("Seeds:"), 4, 0)
        self.eval_seeds_edit = QLineEdit("0,1,2,3,4")
        eval_layout.addWidget(self.eval_seeds_edit, 4, 1)

        layout.addWidget(eval_group)
        
        # Start evaluation button
        self.eval_button = QPushButton("Start Evaluation")
        self.eval_button.clicked.connect(self.start_evaluation)
        layout.addWidget(self.eval_button)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
    
    def log_message(self, message):
        self.log_text.append(message)
        self.statusBar().showMessage(message.split('\n')[0])
    
    def start_training(self):
        # Parse seeds
        seeds_text = self.seeds_edit.text()
        try:
            seeds = [int(seed.strip()) for seed in seeds_text.split(',')]
        except ValueError:
            self.log_message("Error: Seeds must be comma-separated integers")
            return
        
        # Check if data paths are provided
        train_data_path = self.train_data_path.text()
        valid_data_path = self.valid_data_path.text()
        test_data_path = self.test_data_path.text()
        
        if not train_data_path or not valid_data_path or not test_data_path:
            self.log_message("Error: Please select all required data files")
            return
            
        if not all(os.path.exists(path) for path in [train_data_path, valid_data_path, test_data_path]):
            self.log_message("Error: One or more data files do not exist")
            return
        
        # Collect configuration
        config = {
            # 'universe': self.universe_combo.currentText(),
            'universe': 'csi300',
            # 'prefix': self.prefix_combo.currentText(),
            'prefix': 'opensource',
            'train_data_path': train_data_path,
            'valid_data_path': valid_data_path,
            'test_data_path': test_data_path,
            'd_feat': self.d_feat_spin.value(),
            'd_model': self.d_model_spin.value(),
            't_nhead': self.t_nhead_spin.value(),
            's_nhead': self.s_nhead_spin.value(),
            'dropout': self.dropout_spin.value(),
            'gate_input_start_index': self.gate_start_spin.value(),
            'gate_input_end_index': self.gate_end_spin.value(),
            'beta': self.beta_spin.value(),
            'n_epoch': self.epoch_spin.value(),
            'lr': self.lr_spin.value(),
            'GPU': self.gpu_spin.value(),
            'train_stop_loss_thred': self.stop_loss_spin.value(),
            'seeds': seeds
        }
        
        # Create worker thread
        self.worker = TrainingWorker(config)
        self.worker.update_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.training_finished)
        
        # Update UI
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_message("Starting training with configuration:")
        for key, value in config.items():
            self.log_message(f"  {key}: {value}")
        
        # Start worker
        self.worker.start()
    
    def stop_training(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.log_message("Training stopped by user")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def training_finished(self, results):
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Log results
        self.log_message("\nTraining Results:")
        self.log_message(f"IC: {results['IC_mean']:.4f} ± {results['IC_std']:.4f}")
        self.log_message(f"ICIR: {results['ICIR_mean']:.4f} ± {results['ICIR_std']:.4f}")
        self.log_message(f"RIC: {results['RIC_mean']:.4f} ± {results['RIC_std']:.4f}")
        self.log_message(f"RICIR: {results['RICIR_mean']:.4f} ± {results['RICIR_std']:.4f}")
        
        # Update progress to 100%
        self.progress_bar.setValue(100)
    
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "./model",
            "Model Files (*.pkl)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_train_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data File",
            "./opensource",
            "Pickle Files (*.pkl)"
        )
        if file_path:
            self.train_data_path.setText(file_path)
    
    def browse_valid_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Validation Data File",
            "./opensource",
            "Pickle Files (*.pkl)"
        )
        if file_path:
            self.valid_data_path.setText(file_path)
    
    def browse_test_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Data File",
            "./opensource",
            "Pickle Files (*.pkl)"
        )
        if file_path:
            self.test_data_path.setText(file_path)
    
    def browse_eval_test_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Data File",
            "./opensource",
            "Pickle Files (*.pkl)"
        )
        if file_path:
            self.eval_test_data_path.setText(file_path)
    
    def auto_fill_data_paths(self):
        try:
            universe = self.universe_combo.currentText()
            prefix = self.prefix_combo.currentText()
            
            # Auto-fill data paths based on selected universe and prefix
            train_path = f"{prefix}/{universe}_dl_train.pkl"
            valid_path = f"{prefix}/{universe}_dl_valid.pkl"
            test_path = f"{prefix}/{universe}_dl_test.pkl"
            
            # Check if files exist
            if os.path.exists(train_path):
                self.train_data_path.setText(os.path.abspath(train_path))
            else:
                self.log_message(f"Warning: Training data file not found at {train_path}")
                
            if os.path.exists(valid_path):
                self.valid_data_path.setText(os.path.abspath(valid_path))
            else:
                self.log_message(f"Warning: Validation data file not found at {valid_path}")
                
            if os.path.exists(test_path):
                self.test_data_path.setText(os.path.abspath(test_path))
            else:
                self.log_message(f"Warning: Test data file not found at {test_path}")
        except Exception as e:
            # In case of any error during auto-fill
            print(f"Error during auto-fill: {str(e)}")
    
    def update_eval_data_path(self):
        try:
            # universe = self.eval_universe_combo.currentText()
            # prefix = self.eval_prefix_combo.currentText()
            
            # Auto-fill test data path based on selected universe and prefix
            test_path = "opensource/csi300_dl_valid.pkl"
            
            # Check if file exists
            if os.path.exists(test_path):
                self.eval_test_data_path.setText(os.path.abspath(test_path))
            else:
                self.log_message(f"Warning: Test data file not found at {test_path}")
        except Exception as e:
            # In case of any error during auto-fill
            print(f"Error during test data auto-fill: {str(e)}")
    
    def start_evaluation(self):
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(
                self,
                "Invalid Model",
                "Please select a valid model file"
            )
            return
        
        # Get test data path from evaluation tab
        test_data_path = self.eval_test_data_path.text()
        if not test_data_path or not os.path.exists(test_data_path):
            QMessageBox.warning(
                self,
                "Test Data Not Found",
                "Please select a valid test data file"
            )
            return
        
        # universe = self.eval_universe_combo.currentText()
        
        # Get GPU ID and seed from evaluation tab
        gpu_id = self.eval_gpu_spin.value()
        # seed = self.eval_seed_spin.value()
        # Parse seeds
        seeds_text = self.eval_seeds_edit.text()
        try:
            seeds = [int(seed.strip()) for seed in seeds_text.split(',')]
        except ValueError:
            self.log_message("Error: Seeds must be comma-separated integers")
            return
        # Configure model parameters - using same as training tab for consistency except for GPU and seed
        config = {
            'model_path': model_path,
            'test_data_path': test_data_path,
            # 'universe': self.universe_combo.currentText(),
            'universe': 'csi300',
            # 'prefix': self.prefix_combo.currentText(),
            'prefix': 'opensource',
            'd_feat': self.d_feat_spin.value(),
            'd_model': self.d_model_spin.value(),
            't_nhead': self.t_nhead_spin.value(),
            's_nhead': self.s_nhead_spin.value(),
            'dropout': self.dropout_spin.value(),
            'gate_input_start_index': self.gate_start_spin.value(),
            'gate_input_end_index': self.gate_end_spin.value(),
            'beta': self.beta_spin.value(),
            'n_epoch': self.epoch_spin.value(),
            'lr': self.lr_spin.value(),
            'GPU': gpu_id,
            'train_stop_loss_thred': self.stop_loss_spin.value(),
            'seeds': seeds
        }
        
        # Create evaluation worker
        self.evaluation_worker = EvaluationWorker(config)
        self.evaluation_worker.update_signal.connect(self.update_eval_results)
        self.evaluation_worker.finished_signal.connect(self.evaluation_finished)
        
        # Update UI
        self.eval_button.setEnabled(False)
        self.results_text.clear()
        self.log_message("Starting evaluation with configuration:")
        for key, value in config.items():
            self.log_message(f"  {key}: {value}")
        
        # Start evaluation worker
        self.evaluation_worker.start()
    
    def update_eval_results(self, message):
        """Update the evaluation results text area with a message"""
        self.results_text.append(message)
        
    def evaluation_finished(self, metrics):
        """Handle the completion of model evaluation"""
        # Re-enable the evaluation button
        self.eval_button.setEnabled(True)
        
        # Format metrics for display
        self.results_text.append("\nEvaluation Results:")
        self.results_text.append(f"IC: {metrics.get('IC', 0):.4f} ± {metrics.get('IC_std', 0):.4f}")
        self.results_text.append(f"ICIR: {metrics.get('ICIR', 0):.4f} ± {metrics.get('ICIR_std', 0):.4f}")
        self.results_text.append(f"RIC: {metrics.get('RIC', 0):.4f} ± {metrics.get('RIC_std', 0):.4f}")
        self.results_text.append(f"RICIR: {metrics.get('RICIR', 0):.4f} ± {metrics.get('RICIR_std', 0):.4f}")
        
        # Additional metrics if available
        for key, value in metrics.items():
            if key not in ['IC', 'ICIR', 'RIC', 'RICIR', 'IC_std', 'ICIR_std', 'RIC_std', 'RICIR_std', 'seeds_used']:
                self.results_text.append(f"{key}: {value}")
                
        # Log a summary message to the status bar
        self.statusBar().showMessage(f"Evaluation complete - IC: {metrics.get('IC', 0):.4f}, ICIR: {metrics.get('ICIR', 0):.4f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MASTERModelUI()
    window.show()
    sys.exit(app.exec())
