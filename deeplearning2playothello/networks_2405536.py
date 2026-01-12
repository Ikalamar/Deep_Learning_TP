import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
    
    

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        
        #1st option: using hidden states
        # self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        #2nd option: using output seauence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        #1st option: using hidden states as below
        # outp = self.hidden2output(torch.cat((hn,cn),-1))
        
        #2nd option: using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class CNN(nn.Module):
    def __init__(self, conf):
        """
        Convolutional Neural Network (CNN) model for the Othello game.
        Optimized architecture for 8x8 board state prediction.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(CNN, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_input_seq = conf["len_inpout_seq"]
        
        # Optimized CNN architecture for Othello
        # Input: (batch_size, 1, 8, 8) - single board or (batch_size, seq_len, 8, 8)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual-like connection for deeper learning
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Global average pooling + fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, self.board_size * self.board_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, seq):
        """
        Forward pass of the CNN.

        Parameters:
        - seq (torch.Tensor): Board state(s) as input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        # Handle different input shapes (work with torch tensors, avoid np.squeeze)
        if len(seq.shape) == 4:
            # Batch of sequences: (batch_size, seq_len, 8, 8)
            # Take only the last board state in sequence
            seq = seq[:, -1, :, :].unsqueeze(1)  # (batch_size, 1, 8, 8)
        elif len(seq.shape) == 3:
            # Single sequence: (seq_len, 8, 8)
            seq = seq[-1, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        elif len(seq.shape) == 2:
            # Single board: (8, 8)
            seq = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        
        # Convolutional layers with batch normalization and ReLU activation
        x = self.relu(self.bn1(self.conv1(seq)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        """
        Train the CNN model with early stopping.
        """
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        
        best_dev = 0.0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + 
                  str(loss_batch / nb_batch))
            last_training = time.time() - start_time
            
            self.eval()
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100*acc_train, 2)}%, Dev:{round(100*acc_dev, 2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")
            
            # Save model for every epoch
            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            
            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
            
            print("*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev, 3)}%")
        
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    def evalulate(self, test_loader, device):
        """
        Evaluate the CNN model on test data.
        """
        all_predicts = []
        all_targets = []
        
        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
        
        perf_rep = classification_report(all_targets,
                                        all_predicts,
                                        zero_division=1,
                                        digits=4,
                                        output_dict=True)
        
        return perf_rep


class CNNBasic(nn.Module):
    def __init__(self, conf):
        """
        Basic Convolutional Neural Network (CNN) model for the Othello game.
        Simple architecture without optimizations (no batch norm, no dropout).

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(CNNBasic, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_input_seq = conf["len_inpout_seq"]
        
        # Basic CNN architecture for Othello
        # Input: (batch_size, 1, 8, 8) - single board
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers (simple)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, self.board_size * self.board_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, seq):
        """
        Forward pass of the basic CNN.

        Parameters:
        - seq (torch.Tensor): Board state(s) as input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        # Handle different input shapes (work with torch tensors, avoid np.squeeze)
        if len(seq.shape) == 4:
            # Batch of sequences: (batch_size, seq_len, 8, 8)
            # Take only the last board state in sequence
            seq = seq[:, -1, :, :].unsqueeze(1)  # (batch_size, 1, 8, 8)
        elif len(seq.shape) == 3:
            # Single sequence: (seq_len, 8, 8)
            seq = seq[-1, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        elif len(seq.shape) == 2:
            # Single board: (8, 8)
            seq = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        
        # Convolutional layers with ReLU activation (no batch norm, no dropout)
        x = self.relu(self.conv1(seq))
        x = self.relu(self.conv2(x))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        """
        Train the basic CNN model with early stopping.
        """
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        
        best_dev = 0.0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + 
                  str(loss_batch / nb_batch))
            last_training = time.time() - start_time
            
            self.eval()
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100*acc_train, 2)}%, Dev:{round(100*acc_dev, 2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")
            
            # Save model for every epoch
            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            
            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
            
            print("*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev, 3)}%")
        
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    def evalulate(self, test_loader, device):
        """
        Evaluate the basic CNN model on test data.
        """
        all_predicts = []
        all_targets = []
        
        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
        
        perf_rep = classification_report(all_targets,
                                        all_predicts,
                                        zero_division=1,
                                        digits=4,
                                        output_dict=True)
        
        return perf_rep


class CNNLSTM(nn.Module):
    def __init__(self, conf):
        """
        Hybrid CNN-LSTM model for the Othello game.
        Combines CNN for spatial feature extraction with LSTM for temporal sequence modeling.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(CNNLSTM, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_input_seq = conf["len_inpout_seq"]
        
        # CNN component for spatial feature extraction
        # Input: (batch_size, seq_len, 8, 8)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # CNN output: (batch_size, 64, 8, 8) -> flattened: (batch_size, 64*8*8=4096)
        cnn_output_dim = 64 * 8 * 8  # 4096
        
        # LSTM component for temporal sequence modeling
        # Input: (batch_size, seq_len, cnn_output_dim)
        self.lstm_hidden_dim = 256
        self.lstm = nn.LSTM(cnn_output_dim, self.lstm_hidden_dim, 
                           num_layers=2, batch_first=True, dropout=0.2)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(self.lstm_hidden_dim, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.board_size * self.board_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, seq):
        """
        Forward pass of the CNN-LSTM model.

        Parameters:
        - seq (torch.Tensor): Sequence of board states as input.
                              Shape: (batch_size, seq_len, 8, 8) or (seq_len, 8, 8) or (8, 8)

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        # Handle different input shapes
        if len(seq.shape) == 4:
            # Batch of sequences: (batch_size, seq_len, 8, 8)
            batch_size, seq_len, h, w = seq.shape
        elif len(seq.shape) == 3:
            # Single sequence: (seq_len, 8, 8)
            seq = seq.unsqueeze(0)  # Add batch dimension: (1, seq_len, 8, 8)
            batch_size, seq_len, h, w = seq.shape
        elif len(seq.shape) == 2:
            # Single board: (8, 8)
            seq = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
            batch_size, seq_len, h, w = 1, 1, 8, 8
        
        # Reshape for CNN: (batch_size*seq_len, 1, 8, 8)
        seq_reshaped = seq.reshape(batch_size * seq_len, 1, h, w)
        
        # CNN forward pass
        x = self.relu(self.bn1(self.conv1(seq_reshaped)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Flatten CNN output: (batch_size*seq_len, 64*8*8)
        x = x.reshape(batch_size * seq_len, -1)
        
        # Reshape back for LSTM: (batch_size, seq_len, 64*8*8)
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the last output from LSTM: (batch_size, lstm_hidden_dim)
        lstm_last = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(lstm_last))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        """
        Train the CNN-LSTM model with early stopping.
        """
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        
        best_dev = 0.0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            
            self.train()
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + 
                  str(loss_batch / nb_batch))
            last_training = time.time() - start_time
            
            self.eval()
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100*acc_train, 2)}%, Dev:{round(100*acc_dev, 2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")
            
            # Save model for every epoch
            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            
            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
            
            print("*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev, 3)}%")
        
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    def evalulate(self, test_loader, device):
        """
        Evaluate the CNN-LSTM model on test data.
        """
        all_predicts = []
        all_targets = []
        
        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
        
        perf_rep = classification_report(all_targets,
                                        all_predicts,
                                        zero_division=1,
                                        digits=4,
                                        output_dict=True)
        
        return perf_rep
            

