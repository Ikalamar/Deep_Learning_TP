import torch
from torch.utils.data import DataLoader

from data import CustomDatasetMany
from utile import BOARD_SIZE
from networks_2405536 import LSTMs


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
print('Running on ' + str(device))

len_samples=5

dataset_conf={}  
# self.filelist : a list of all games for train/dev/test
dataset_conf["filelist"]="train.txt"
#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
dataset_conf["len_samples"]=len_samples
dataset_conf["path_dataset"]="../dataset/"
dataset_conf['batch_size']=512

print("Training Dataste ... ")
ds_train = CustomDatasetMany(dataset_conf)
trainSet = DataLoader(ds_train, batch_size=dataset_conf['batch_size'])

dataset_conf={}  
# self.filelist : a list of all games for train/dev/test
dataset_conf["filelist"]="dev.txt"
#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
dataset_conf["len_samples"]=len_samples
dataset_conf["path_dataset"]="../dataset/"
dataset_conf['batch_size']=256

print("Development Dataste ... ")
ds_dev = CustomDatasetMany(dataset_conf)
devSet = DataLoader(ds_dev, batch_size=dataset_conf['batch_size'])

conf={}
conf["board_size"]=BOARD_SIZE
conf["path_save"]="save_models_LSTM_DO_0-2_Bidirect"
conf['epoch']=50
conf["earlyStopping"]=25
conf["len_inpout_seq"]=len_samples
conf["LSTM_conf"]={}
conf["LSTM_conf"]["hidden_dim"]=128
conf["LSTM_conf"]["dropout"] = 0.2
conf["LSTM_conf"]["bidirectionnal"]=True

model = LSTMs(conf).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.005)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

best_epoch=model.train_all(trainSet,
                       devSet,
                       conf['epoch'],
                       device, opt)

model = torch.load(conf["path_save"] + '/model_2.pt')
model.eval()
train_clas_rep=model.evalulate(trainSet, device)
acc_train=train_clas_rep["weighted avg"]["recall"]
print(f"Accuracy Train:{round(100*acc_train,2)}%")

