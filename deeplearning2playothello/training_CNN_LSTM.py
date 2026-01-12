import torch
from torch.utils.data import DataLoader

from data import CustomDatasetOne
from utile import BOARD_SIZE
from networks_2405536 import CNNLSTM


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
print('Running on ' + str(device))

len_samples = 5  # Using sequence of 5 board states for LSTM

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
dataset_conf = {}  
dataset_conf["filelist"] = "train.txt"
dataset_conf["len_samples"] = len_samples
dataset_conf["path_dataset"] = "../dataset/"
dataset_conf['batch_size'] = 256

print("Loading Training Dataset ... ")
ds_train = CustomDatasetOne(dataset_conf, load_data_once4all=True)
trainSet = DataLoader(ds_train, batch_size=dataset_conf['batch_size'], shuffle=True)

# ============================================================================
# DEVELOPMENT CONFIGURATION
# ============================================================================
dataset_conf = {}  
dataset_conf["filelist"] = "dev.txt"
dataset_conf["len_samples"] = len_samples
dataset_conf["path_dataset"] = "../dataset/"
dataset_conf['batch_size'] = 256

print("Loading Development Dataset ... ")
ds_dev = CustomDatasetOne(dataset_conf, load_data_once4all=True)
devSet = DataLoader(ds_dev, batch_size=dataset_conf['batch_size'])

# ============================================================================
# CNN-LSTM MODEL CONFIGURATION - DEFAULT HYPERPARAMETERS
# ============================================================================
conf = {}
conf["board_size"] = BOARD_SIZE
conf["path_save"] = "save_models_CNN_LSTM"
conf['epoch'] = 20
conf["earlyStopping"] = 20
conf["len_inpout_seq"] = len_samples
conf["CNNLSTM_conf"] = {}

# Initialize model
model = CNNLSTM(conf).to(device)

# Default optimizer settings
opt = torch.optim.Adam(model.parameters(), lr=0.001)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print(f"Number of trainable parameters: {n}")
print("=" * 60)

# Train the model
print("\nStarting training with CNN-LSTM architecture...")
best_epoch = model.train_all(trainSet,
                             devSet,
                             conf['epoch'],
                             device, 
                             opt)

# ============================================================================
# EVALUATION ON TRAINING SET
# ============================================================================
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

model = torch.load(conf["path_save"] + '_CNN_LSTM/model_' + str(best_epoch) + '.pt', weights_only=False)
model.eval()

train_clas_rep = model.evalulate(trainSet, device)
acc_train = train_clas_rep["weighted avg"]["recall"]
print(f"Final Accuracy on Training Set: {round(100*acc_train, 2)}%")

dev_clas_rep = model.evalulate(devSet, device)
acc_dev = dev_clas_rep["weighted avg"]["recall"]
print(f"Final Accuracy on Development Set: {round(100*acc_dev, 2)}%")

print("=" * 60)
print(f"Best model saved at epoch: {best_epoch}")
print(f"Model path: {conf['path_save']}_CNN_LSTM/model_{best_epoch}.pt")
print("=" * 60)
