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


if __name__ == "__main__":
    #Setting Parameters
    parser = argparse.ArgumentParser(description='EEGTP')
    parser.add_argument('--paths', type=str, default='/Users/simon/ky/EEG/Data/*.mat', metavar='N', help='directory name (default: paths)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 60)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_epochs_pretrain', type=int, default=100, metavar='N', help='number of epochs to pretrain (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--input_dim', type=int, default=1, metavar='N', help='dimension of imput space (default: 1)')
    #parser.add_argument('--latent_dim', type=int, default=136, metavar='N', help='dimension of latent space (default: 64)')
    #parser.add_argument('--h_dim', type=int, default=1000, metavar='N', help='dimension of hidden space (default: 128)')
    parser.add_argument('--d_model', type=int, default=512, metavar='N', help='dimension of embedding(default: 512)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='N', help='number of layers (default: 2)')
    parser.add_argument('--head', type=int, default=8, metavar='N', help='head of attention (default: 8)')
    parser.add_argument('--start', type=int, default=296, metavar='N', help='size of train_dataset (default: 200)')
    parser.add_argument('--end', type=int, default=300, metavar='N', help='size of train_dataset (default: 300)')
    parser.add_argument('--tw', type=int, default=6800, metavar='N', help='time windom (default: 6801*3)')
    parser.add_argument('--h_dim', type=int, default=3400, metavar='N', help='dimension of h0 (default: 100)')
    parser.add_argument('--pred', type=int, default=6800, metavar='N', help='length of stride (default: 6801)')
    parser.add_argument('--rank', type=int, default=50, metavar='N', help='dimension of attention layers (default: 128)')
    parser.add_argument('--early_stop', type=int, default=5, metavar='N', help='number of training without improving (default: 5)')
    args = parser.parse_args()

    model_dict = init_model_dict(args.input_dim,args.h_dim,args.d_model, args.num_layers,args.tw,args.pred,args.head,args.rank)
    #print(model_dict)
    for m in model_dict:
        model_dict[m].to(device)
        
    ###########Pretraining#############
    #pretrain_g = torch.cat((train_g,test_g,val_g),0)
    Pretrain_Dataloader = DataLoader(gan_data, batch_size=args.batch_size, shuffle=False)
    optim_dict = init_optim_dict(args.learning_rate)
    Loss_D_pt =[]
    Loss_G_pt = []
    for epoch in range(args.num_epochs_pretrain+1):
        loss_dict = {'G': [], 'D': []}
        mean_loss_D, mean_loss_G = 0, 0
        for y in tqdm(Pretrain_Dataloader,desc = f'[train]epoch:{epoch}'):
            batch_size=y.shape[0]
            y = y.to(device)
            loss_D, loss_G, fake_data =pretrain_epoch(optim_dict,model_dict,y,args.d_model,args.pred,args.tw)
            loss_dict["D"].append(loss_D.detach().item())
            loss_dict["G"].append(loss_G.detach().item())
        mean_loss_D = sum(loss_dict["D"])/len(loss_dict["D"])
        mean_loss_G = sum(loss_dict["G"])/len(loss_dict["G"])
        Loss_D_pt.append(mean_loss_D)
        Loss_G_pt.append(mean_loss_G)
        print(f'[{epoch+1}/{args.num_epochs_pretrain}] Pretraining: Loss_D: {mean_loss_D} Loss_G: {mean_loss_G}')
    
        if epoch % 5 ==0:
            fig, ax = plt.subplots(figsize=[25, 5])
            ax.plot(y[:,-args.pred:,:].cpu().detach().numpy()[-1].reshape(-1, 1)[-args.pred//100:])
            ax.plot(fake_data.cpu().detach().numpy()[-1].reshape(-1, 1)[-args.pred//100:])
            plt.legend(['Orig', 'Gen'])
            plt.title('Predictions Compares')
            plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.plot(Loss_G_pt, color = 'blue', label = 'Generator Loss')
    plt.plot(Loss_D_pt, color = 'black', label = 'Discriminator Loss')
    plt.title('AttWGAN-gp Loss')
    plt.xlabel('epochs')
    plt.legend(loc = 'upper right')
    
    ###Training
    Train_Dataloader = DataLoader(TensorDataset(train_x, train_h, train_g),batch_size=args.batch_size,shuffle=False,)
    Val_Dataloader = DataLoader(TensorDataset(val_x, val_h, val_g),  batch_size = args.batch_size, shuffle = False,)
    Test_Dataloader = DataLoader(TensorDataset(test_x, test_h, test_g),batch_size=args.batch_size,shuffle=False,)
    optim_dict = init_optim_dict(args.learning_rate)
    best_loss, early_stop_count = math.inf, 0
    Loss_tr_epoch = []
    Loss_val_epoch = []
    Loss_te_epoch = []
    for epoch in range(args.num_epochs+1):
        loss_tr = []
        mean_loss_tr = 0
        train_steps = len(Train_Dataloader)
        for (x, h, y) in tqdm(Train_Dataloader,desc = f'[train]epoch:{epoch}'):
            batch_size=x.shape[0]
            h, x ,y = h.to(device), x.to(device), y.to(device)
            loss_eg, recon_attn, prediction = train_epoch(optim_dict,model_dict,x,h,y, loss_MSE,loss_KLD,args.tw, args.pred)
            loss_tr.append(loss_eg.detach().item())
            
        mean_loss_tr = sum(loss_tr)/len(loss_tr)
    
        Loss_tr_epoch.append(mean_loss_tr)
        print(f'[{epoch+1}/{args.num_epochs}] Training: Toal_loss: {mean_loss_tr}')
        ##Validating
        for m in model_dict:
            model_dict[m].eval() ##!!!!
        loss_val = []
        V_loss = []
        mean_loss_val = 0
        mean_V_loss = 0
        for (x,h,y) in tqdm(Val_Dataloader):
            batch_size=x.shape[0]
            h, x ,y = h.to(device), x.to(device), y.to(device)
            bs= x.size(0)
            out_E = model_dict["E"](x,h)  
            recon_out_attn = model_dict["DA"](out_E) 
            g_out = model_dict["G"](out_E)
            pred_val = model_dict["F"](g_out)##mean
            loss_re_attn = loss_MSE(recon_out_attn, x)  
            #loss_norm_attn = loss_KLD(out_E[0], out_E[1]) 
            loss_eg = loss_MSE(pred_val, y)
            V_loss_attn = loss_re_attn 
            V_loss.append(V_loss_attn.detach().item()) ##
            loss_val.append(loss_eg.detach().item()) ##
        mean_V_loss = sum(V_loss)/len(V_loss)
        mean_loss_val = sum(loss_val)/len(loss_val)
        mean_both_loss = mean_V_loss + mean_loss_val
        Loss_val_epoch.append(mean_loss_val)
        #Plotting
        if epoch % 5 ==0:
            fig, ax = plt.subplots(figsize=[25, 5])
            ax.plot(y.cpu().numpy()[-1].reshape(-1, 1)[:args.pred//10])
            ax.plot(pred_val.cpu().detach().numpy()[-1].reshape(-1, 1)[:args.pred//10])
            plt.legend(['Orig', 'Pred_val'])
            plt.title('Compare prediction')
            plt.show()
        if epoch % 5 ==0:
            fig, ax = plt.subplots(figsize=[25, 5])
            ax.plot(y.cpu().numpy()[-1].reshape(-1, 1)[args.pred//9:args.pred//5])
            ax.plot(pred_val.cpu().detach().numpy()[-1].reshape(-1, 1)[args.pred//9:args.pred//5])
            plt.legend(['Orig', 'Pred_val'])
            plt.title('Compare prediction')
            plt.show()
        if epoch % 5 ==0:
            fig, ax = plt.subplots(figsize=[25, 5])
            ax.plot(y.cpu().numpy()[-1].reshape(-1, 1)[args.pred//4:args.pred//3])
            ax.plot(pred_val.cpu().detach().numpy()[-1].reshape(-1, 1)[args.pred//4:args.pred//3])
            plt.legend(['Orig', 'Pred_val'])
            plt.title('Compare prediction')
            plt.show()
        if epoch % 5 ==0:
            fig, ax = plt.subplots(figsize=[25, 5])
            ax.plot(y.cpu().numpy()[-1].reshape(-1, 1)[-args.pred//10:])
            ax.plot(pred_val.cpu().detach().numpy()[-1].reshape(-1, 1)[-args.pred//10:])
            plt.legend(['Orig', 'Pred_val'])
            plt.title('Compare prediction')
            plt.show()
        ##saving model
        if mean_both_loss < best_loss:
            best_loss = mean_both_loss
            early_stop_count = 0
            for fname, module in model_dict.items():
                full_path = './' + fname + ".pth"
                torch.save(module.state_dict(), pjoin(full_path))
            print(f'Saving model with loss :{best_loss:.4f}')
        else:
            early_stop_count += 1
        print(early_stop_count)
        print(args.early_stop)
        if early_stop_count >= args.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            break
        ##Testing
        for m in model_dict:
            model_dict[m].eval() ##!!!!
        Te_loss = []
        mean_loss_test = 0
        for (x,h,y) in tqdm(Test_Dataloader):
            batch_size=x.shape[0]
            h, x ,y = h.to(device), x.to(device), y.to(device)
            out_E = model_dict["E"](x)
            pred_te =model_dict["G"](out_E) ##mean
            loss_VG = loss_MSE(pred_te, y)
            Te_loss.append(loss_VG.detach().item())
        mean_loss_test = sum(Te_loss)/len(Te_loss)
        Loss_te_epoch.append(mean_loss_test)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, mean_loss_tr, mean_loss_val, mean_loss_test))
        Loss_te_epoch.append(mean_loss_test)
