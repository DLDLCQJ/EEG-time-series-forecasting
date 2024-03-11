import os
from os.path import dirname, join as pjoin
from glob import glob

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.io as sio
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import torch
from torch import nn, optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
#from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error
import math

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated
device =torch.device('cuda' if torch.cuda.is_available() else 'mps')#训练设备
print(device)

batch_size = 60
learning_rate = 0.000115
num_epochs = 100

##VAE
latent_dim =64 ##
hidden_dim = 64*2 
input_dim= output_dim = gen_input_dim = 1 
num_layers =1


#trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size = batch_size, shuffle = False)
trainDataloader = DataLoader(TensorDataset(train_x_slide, h0_tr, train_y_gan),batch_size=batch_size,shuffle=True,)
for batch_idx, (x, y, g) in enumerate(trainDataloader):
    print(x.shape, y.shape, g.shape)
    ###########Starting#############
model_dict = init_model_dict(input_dim,hidden_dim,latent_dim,num_layers,gen_input_dim)
#print(model_dict)
for m in model_dict:
    model_dict[m].to(device)
histV = np.zeros(num_epochs+1)
histG = np.zeros(num_epochs+1)
histD = np.zeros(num_epochs+1)
histEG = np.zeros(num_epochs+1)

#%%
###########Pretraining#############
optim_dict = init_optim_dict(learning_rate)
for epoch in range(num_epochs+1):
    loss_G=0
    loss_D=0
    
    for (x, h0_tr, y) in tqdm(trainDataloader,desc = f'[train]epoch:{epoch}'):
        batch_size=y.shape[0]
        x = x.to(device) #
        y = y.to(device)
        
        fake_data, loss_dict=pretrain_epoch(optim_dict,model_dict,y,loss_BCE,latent_dim)
        loss_G =loss_dict["G"][0]
        loss_D =loss_dict["D"][0]
        
     
    histG[epoch] = loss_G
    histD[epoch] = loss_D
    print(f'[{epoch+1}/{num_epochs}] LossG: {histG[epoch]} LossD: {histD[epoch]}')

    if epoch % 5 ==0:
        fig, ax = plt.subplots(figsize=[25, 5])
        ax.plot(y[:, 6801:, :].cpu().detach().numpy()[-1].reshape(2267, 1))
        ax.plot(fake_data[:, 6801:, :].cpu().detach().numpy()[-1].reshape(2267, 1))
        plt.legend(['Orig', 'Gen'])
        plt.title('Compare prediction')
        plt.show()

plt.figure(figsize = (12, 6))
plt.plot(histG, color = 'blue', label = 'Generator Loss')
plt.plot(histD, color = 'black', label = 'Discriminator Loss')
plt.title('WGAN-gp Loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')


#%%
###########Training#############
optim_dict = init_optim_dict(learning_rate)
for epoch in range(num_epochs+1):
    loss_V=0
    loss_EG=0
    num_samples=0
    for (x, h0_tr, y) in tqdm(trainDataloader,desc = f'[train]epoch:{epoch}'):
        batch_size=y.shape[0]
        x = x.to(device)
        h0_tr = h0_tr.unsqueeze(0).to(device) #[1, 60, 128]
        y = y.to(device)
        

        loss_dict, prediction = train_epoch(optim_dict,model_dict,x,h0_tr,y,loss_MSE,loss_KLD,latent_dim)
  
        loss_V =loss_dict["V"][0]
        loss_EG =loss_dict["EG"][0]
        
    histV[epoch] = loss_V
    histEG[epoch] = loss_EG
    print(f'[{epoch+1}/{num_epochs}] LossV: {histV[epoch]} LossEG: {histEG[epoch]}')
    if epoch % 5 ==0:
        fig, ax = plt.subplots(figsize=[25, 5])
        ax.plot(y[:, 6801:, :].cpu().detach().numpy()[-1].reshape(2267, 1))
        ax.plot(prediction[:, 6801:, :].cpu().detach().numpy()[-1].reshape(2267, 1))
        plt.legend(['Orig', 'Pred'])
        plt.title('Compare prediction')
        plt.show()

#%%
plt.figure(figsize = (12, 6))
plt.plot(histV, color = 'grey', label = 'VAE Loss')
plt.plot(histEG, color = 'red', label = 'Finalmodel Loss')
plt.title('VAEGAN Loss')
plt.xlabel('times')
plt.legend(loc = 'upper right')

#%%
#########Predicting############
h0_te = net(test_xt.to(device))
print(h0_te.shape)
##
#trainDataloader = DataLoader(TensorDataset(train_x_slide, h0_tr, train_y_gan), shuffle = False, batch_size = 1,)
testDataloader = DataLoader(TensorDataset(test_x_slide, h0_te, test_y_gan), shuffle = False, batch_size = 60, )

with torch.no_grad():
    for m in model_dict:
        model_dict[m].eval() ##!!!!

pred_y_train = []     
for (x,h0_tr,y) in tqdm(trainDataloader,desc = f'[train]epoch:{epoch}'):
    x = x.to(device)
    h0_tr = h0_tr.unsqueeze(0).to(device) #[1, 60, 128]
    y = y.to(device)
    pred_tr_data = model_dict["G"](model_dict["V"].encoder(x, h0_tr)[1].reshape(-1,latent_dim,1))
    pred_y_train.append(pred_tr_data.reshape(-1, 6801//3, 1).detach())


pred_y_test = []     
for (x,h0_te,y) in tqdm(testDataloader,desc = f'[train]epoch:{epoch}'):
    x = x.to(device)
    h0_te = h0_te.unsqueeze(0).to(device) #[1, 60, 128]
    y = y.to(device)
    pred_te_data = model_dict["G"](model_dict["V"].encoder(x, h0_te)[1].reshape(-1,latent_dim,1))
    pred_y_test.append(pred_te_data.reshape(-1, 6801//3, 1).detach())
print(pred_y_train[-1].dtype)
print(pred_y_train[-1].shape)
print(train_yt.shape)

#%%
from sklearn.metrics import mean_squared_error
import math
fig, axis = plt.subplots(2,1,figsize=(12, 8))

axis[0].plot(train_yt[-1], label = 'True_tr value')
axis[0].plot(torch.squeeze(torch.cat(pred_y_train)).cpu().detach().numpy()[-1], label = 'Predict_tr value')
axis[0].set_title('VAEWGAN true & prediction in train')
axis[1].plot(test_yt[-1],  label = 'True_te value')
axis[1].plot(torch.squeeze(torch.cat(pred_y_test)).cpu().detach().numpy()[-1], label = 'Predict_te value')
axis[1].set_title('VAEWGAN true & prediction in test')
for ax in axis.flat:
    ax.set(xlabel = 'Times', ylabel = 'Amplitude',)
    leg = ax.legend(loc = 'upper right')
plt.show()
MSE_te = mean_squared_error(test_yt, torch.squeeze(torch.cat(pred_y_test)).cpu().detach().numpy())
RMSE_te = math.sqrt(MSE_te)
print(f'Testing RMSE:{RMSE_te}')


y_train_true = y_scaler.inverse_transform(train_yt)
print(torch.cat(pred_y_train).shape)
y_train_pred = y_scaler.inverse_transform(torch.squeeze(torch.cat(pred_y_train)).cpu().detach().numpy())

y_test_true = y_scaler.inverse_transform(test_yt)
y_test_pred = y_scaler.inverse_transform(torch.squeeze(torch.cat(pred_y_test)).cpu().detach().numpy())

plt.figure(figsize=(12, 8))
plt.plot(y_train_true[-1,:], color='black', label = 'True_tr value')
plt.plot(y_train_pred[-1,:], color='blue', label = 'Predict_tr value')
plt.plot(y_test_true[-1,:],  label = 'True_te value')
plt.plot(y_test_pred[-1,:],  label = 'Predict_te value')
plt.title('VAEWGAN true & prediction')
plt.ylabel('Amplitude')
plt.xlabel('times')
plt.legend(loc = 'upper right')

MSE_te = mean_squared_error(y_test_true, y_test_pred)
RMSE_te = math.sqrt(MSE_te)
print(f'Testing RMSE:{RMSE_te}')
# %%
