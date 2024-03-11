
import os
from os.path import dirname, join as pjoin
from glob import glob

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def split_tr_te(dataset):
    data_xl =[]
    data_yl =[]
    data_gl =[]
    for data in tqdm(dataset):
        ##
        data_xf = data.flatten()#.reshape(-1,1)
        print(data_xf.shape)
        ##
        data_xr_slide, data_yr_slide, data_gr_slide = EEG_sliding_window(data_xf, tw, step)


        data_xl.append(data_xr_slide)
        data_yl.append(data_yr_slide)
        data_gl.append(data_gr_slide)
   
    #return train_xl, train_yl, train_gl, test_xl, test_yl, test_gl
    return np.squeeze(np.array(data_xl)), np.squeeze(np.array(data_yl)), np.squeeze(np.array(data_gl))

def Loading_data(paths):
    eeg_list=[]
    for path in paths:
        mat_data = sio.loadmat(path)
        #print(mat_data['EpochData'][0][0][:564,:].shape)
        eeg_list.append(mat_data['EpochData'][0][0][:564,:])
    
    split = int(len(eeg_list)* 0.8)
    train_x, test_x = eeg_list[: split], eeg_list[split:]
    return train_x, test_x

def EEG_sliding_window(x, tw, step):
    x_ = []
    y_ = []
    y_gan = []
    L = len(x)
    for i in range(tw+step, L, step):
        tmp_x = x[i-step-tw: i-tw]
        tmp_y = x[i-tw:i]
        tmp_y_gan = np.concatenate((tmp_x, tmp_y),axis=0)
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    
    return x_, y_, y_gan


paths = glob(r'/Users/simon/ky/EEG/Data/*.mat')
paths = paths[:3]

tw = 6801//3
step=6801
train_dataset,test_dataset = Loading_data(paths)
train_x, train_y, train_g= split_tr_te(train_dataset)
test_x, test_y, test_g = split_tr_te(test_dataset)

print(f'train_x: {train_x.shape} train_y: {train_y.shape} train_g: {train_g.shape}')
print(f'test_x: {test_x.shape} test_y: {test_y.shape} test_g: {test_g.shape}')

train_xr = train_x.reshape(-1,6801)
test_xr = test_x.reshape(-1,6801)
train_yr = train_y.reshape(-1,2267)
test_yr = test_y.reshape(-1,2267)
train_gr = train_g.reshape(-1,9068)
test_gr = test_g.reshape(-1,9068)
##
x_scaler = MinMaxScaler(feature_range = (0, 1))    
train_xs = x_scaler.fit_transform(train_xr)
test_xs = x_scaler.transform(test_xr)
y_scaler =MinMaxScaler(feature_range = (0, 1)) 
train_ys = y_scaler.fit_transform(train_yr)
test_ys = y_scaler.transform(test_yr)
g_scaler =MinMaxScaler(feature_range = (0, 1)) 
train_gs = g_scaler.fit_transform(train_gr)
test_gs = g_scaler.transform(test_gr)
print(train_xs.shape)
print(train_gs.shape)
##
train_xt = torch.from_numpy(train_xs).float()
train_yt = torch.from_numpy(train_ys).float()
train_gt = torch.from_numpy(train_gs).float()
test_xt = torch.from_numpy(test_xs).float()
test_yt = torch.from_numpy(test_ys).float()
test_gt = torch.from_numpy(test_gs).float()
##
train_x_slide = torch.unsqueeze(train_xt,2)
test_x_slide = torch.unsqueeze(test_xt,2)
train_y_slide = torch.unsqueeze(train_yt,2)
test_y_slide = torch.unsqueeze(test_yt,2)
train_y_gan = torch.unsqueeze(train_gt,2)
test_y_gan = torch.unsqueeze(test_gt,2)

n_seq,seq_len,n_features = train_x_slide.shape
print(f'train_x_slide: {train_x_slide.shape} train_y_slide: {train_y_slide.shape} train_y_gan: {train_y_gan.shape}')
print(f'test_x_slide: {test_x_slide.shape} test_y_slide: {test_y_slide.shape} test_y_gan: {test_y_gan.shape}')

