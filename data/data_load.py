
import os
from os.path import dirname, join as pjoin
from glob import glob

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class Dataset_EEG_Data():
    def __init__(self, flag='train', size=None, 
                 path='/Users/simon/ky/EEG/Data/*.mat',features='S', target='EpochData'):
      
        self.seq_len = size[0]
        self.label_len = size[1] #label_len
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val','pred']
        self.flag = flag
        
        self.features = features
        self.target = target
        self.path = path
        
    def read_data(self):
        df_list = []
        paths = glob(self.path)
        #path = paths[1:2]
        #(path)
        for file in paths:
            mat_data = sio.loadmat(file)
            df_list.append(mat_data[self.target][0][0][:564, :6800]) 
        return df_list
    
    def split_train_data(self,df_list):
        for df_raw in df_list:
            df_raw = np.array(df_raw).reshape(-1, 1)
            ##
            df_data = df_raw if self.features == 'S' else df_raw[df_raw.columns[1:]]
            # 分割数据为训练、测试和验证集
            total_len = len(df_data)
            train_end = int((total_len) * 0.5)

            tr_data = df_data[:train_end]
    
            return tr_data

    def split_data(self,df_list):
        for df_raw in df_list:
            df_raw = np.array(df_raw).reshape(-1, 1)
            ##
            df_data = df_raw if self.features == 'S' else df_raw[df_raw.columns[1:]]
            # # 分割数据为训练、测试和验证集
            # total_len = len(df_data)
            # train_end = int((total_len-self.pred_len-self.seq_len) * 0.5)
            # val_end = train_end + int((total_len-self.pred_len-self.seq_len) * 0.25)
            # ##
            # data = df_data[:train_end]
            # if self.flag == 'val':
            #     data = df_data[train_end:val_end]
            # elif self.flag == 'test':
            #     data = df_data[val_end:-(self.pred_len+self.seq_len)]
            # elif self.flag == 'pred':
            #     data = df_data[-(self.pred_len+self.seq_len):]
            # 分割数据为训练、测试和验证集
            total_len = len(df_data)
            train_end = int((total_len) * 0.5)
            val_end = train_end + int((total_len) * 0.25)
            test_end = val_end + int((total_len) * 0.24)
            if self.flag == 'val':
                data = df_data[train_end:val_end]
            elif self.flag == 'test':
                data = df_data[val_end:test_end]
            elif self.flag == 'pred':
                data = df_data[test_end:(test_end+self.pred_len+self.seq_len+1)]
            return data
        
    def slice_data(self, data):
        x_, y_= [], []
        L = len(data)
        for i in range(self.seq_len+self.pred_len, L, self.seq_len):
            tmp_x = data[i-self.pred_len-self.seq_len: i-self.pred_len]
            tmp_y = data[i-self.pred_len:i]
            x_.append(tmp_x)
            y_.append(tmp_y)
        
        return np.array(x_).reshape(-1,self.seq_len), np.array(y_).reshape(-1,self.pred_len)

    def scaler_data(self):
        
        df_list = self.read_data()
        train_data = self.split_train_data(df_list)
        seq_train_x, seq_train_y = self.slice_data(train_data)
        x_scaler = prep.MinMaxScaler(feature_range=(0, 1))
        y_scaler = prep.MinMaxScaler(feature_range=(0, 1))
        seq_train_xs = x_scaler.fit_transform(seq_train_x)
        seq_train_ys = y_scaler.fit_transform(seq_train_y)
        train_xs = torch.from_numpy(seq_train_xs).unsqueeze(-1).float()
        train_ys = torch.from_numpy(seq_train_ys).unsqueeze(-1).float()
        if self.flag != 'train':
            data = self.split_data(df_list)
            seq_x,seq_y = self.slice_data(data)
            seq_xs = x_scaler.transform(seq_x)
            seq_ys = y_scaler.transform(seq_y)
            xs = torch.from_numpy(seq_xs).unsqueeze(-1).float()
            ys = torch.from_numpy(seq_ys).unsqueeze(-1).float()

        if self.flag == 'train':
            return TensorDataset(train_xs,train_ys)
        else:
            return TensorDataset(xs, ys)
