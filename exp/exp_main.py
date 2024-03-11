import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import EEGPT, Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import Dataset, DataLoader
from data.data_loader import Dataset_EEG_Data,Dataset_EEG_Pred
from layers.Convfamily import FeedForwardAdapter,dec_adas,enc_adas
from layers.Attenfamily import Attn_Blocks, AttentionLayer, LoRa_Attn

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

#from layers.Attenfamily import CrossAttn_Adapter
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args,setting):
        super(Exp_Main, self).__init__(args,setting)

    def _build_model(self):
        model_dict = {
            'EEGPT' : EEGPT,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Reformer': Reformer,
        }
        #model = model_dict[self.args.model](self.args).float()
        model = model_dict[self.args.model].Model(self.args).float()
         # ## frozen generator except xxx
        # # for name, param in self.model.named_parameters():
        # #     if 'FeedForwardAdapter' in name:
        # #         param.requires_grad = True
        # #     else:
        # #         param.requires_grad = False

        # ## replacing adapter-attn
        # to_replace = []
        # for name, module in self.model.decoder.named_modules():
        #     if 'Up_Blocks' in name:
        #         to_replace.append(name)
        # for name in to_replace:
        #     new_adapter = model_dict[self.args.model].Adapter(self.args).float()

        #     setattr(model.decoder, name,new_adapter)
        LoRa
        lora_dec = nn.ModuleList(
            [
            Attn_Blocks(AttentionLayer(LoRa_Attn(
                        self.args.d_model, self.args.n_heads, 
                        self.args.pred_len//(self.args.stride_g ** (self.args.g_layers-l-1)),
                        self.args.ranks
                        ),
                         self.args.d_model, self.args.n_heads, lora=True),
                        self.args.d_model
            ) for l in range(self.args.g_layers)
            ])
        lora_enc = nn.ModuleList(
            [
            Attn_Blocks(AttentionLayer(LoRa_Attn(
                        self.args.d_model, self.args.n_heads, 
                        self.args.seq_len//(self.args.stride_e ** (self.args.e_layers-l-1)),
                        self.args.ranks
                        ),
                         self.args.d_model, self.args.n_heads, lora=True),
                        self.args.d_model
            ) for l in range(self.args.e_layers)
            ])


         # reloading pretrain model
        pretraining_path = os.path.join( self.args.checkpoints, self.setting, "Pretrain/best_model.pth")
        model.load_state_dict(torch.load(pretraining_path))
        # adapter
        adapters_dec = nn.ModuleList(
            [
            FeedForwardAdapter(
                self.args.pred_len//(self.args.stride_g ** (self.args.g_layers-l-1)),
                self.args.seq_len//self.args.stride_e**(self.args.e_layers-1),
                self.args.d_model,
            ) for l in range(self.args.g_layers)
            ])
        adapters_enc = nn.ModuleList(
            [
            FeedForwardAdapter(
                self.args.seq_len//(self.args.stride_e**l),
                self.args.seq_len//(self.args.stride_e**l),
                self.args.d_model,
            ) for l in range(self.args.e_layers)
            ])

        ## Enable Fine Tuning
        #dec_adapters(model.decoder, adapters_dec,self.args.d_model, self.args.c_out)
        dec_adas(model.decoder, adapters_dec,self.args.d_model, self.args.c_out)
        enc_adas(model.encoder, adapters_enc)
        ## Enable gradients for new module parameters
        for param in model.parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            #if hasattr(module, 'adapters'):
            if isinstance(module,FeedForwardAdapter):
                for param in module.parameters():
                    param.requires_grad = True
            elif 'projection_final' in name:
                for param in module.parameters():
                    param.requires_grad = True
        
        ## number of parameter
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The model has {total_trainable_params:,} trainable parameters.")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"The model has {total_params:,} total parameters.")  

        ## multi_gpu
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag,scalers=None):
        args = self.args
        data_dict = {
            'EEG':Dataset_EEG_Data
        }
        Data = data_dict[self.args.data]
        data_set = Data(
            path=os.path.join(args.root_path,
                            args.data_path
                            ),
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,
            scalers=scalers
        )
        #data_set = data_processing.scaler_data()
        
        if flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
            Data = Dataset_EEG_Pred
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        print(flag,len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        #model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y):
        # gene input
        # dec_inp = torch.ones_like(batch_y[:,-self.args.pred_len:,:]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        
        #pos_pred = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        #pos_inp = torch.cat([batch_y[:, :self.args.label_len, :], gene_inp], dim=1).float().to(self.device)
        stoken = batch_x[:, :self.args.label_len, :].float().to(self.device)
        #stoken = torch.randn_like(batch_x[:, -self.args.label_len:, :]).float().to(self.device)
        # encoder - decoder
        def _run_model():
            outputs = self.model(batch_x,stoken)
            ##
            # f_dim = -1 if self.args.features == 'MS' else 0
            # for i in range(len(outputs)):
            #     if i < len(outputs)-1:
            #         outputs[i] = outputs[i][:, :self.args.label_len,f_dim:]
            #     else: 
            #         outputs[i] = outputs[i]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:, f_dim:].to(self.device)
        # batch_y = [batch_y.clone() for _ in range(self.args.g_layers)]
        # for i in range(self.args.g_layers):
        #     dim = self.args.pred_len//(self.args.stride_g ** (self.args.g_layers-i-1))
        #     batch_y[i] = batch_y[i][:, :dim, f_dim:].to(self.device)
        return outputs, batch_y

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs, batch_y = self._predict(batch_x, batch_y)

                #for l in range(self.args.g_layers):
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                    #loss += loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        #self.model.train()
        return total_loss


    def train(self, setting):
        train_set, train_loader = self._get_data(flag='train')
        vali_set, vali_loader = self._get_data(flag='val', scalers=(train_set.scaler_x,train_set.scaler_y))
        test_set, test_loader = self._get_data(flag='test', scalers=(train_set.scaler_x,train_set.scaler_y))

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        #path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y)
                #for l in range(self.args.g_layers):
                loss = criterion(outputs,batch_y)
                #loss += loss
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
           
        best_model_path = path + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        for name, param in self.model.named_parameters():
            print(f'Layer: {name}')
            print(param)
        return

    def test(self, setting, test=False):
        train_set, train_loader = self._get_data(flag='train')
        test_set, test_loader = self._get_data(flag='test', scalers=(train_set.scaler_x,train_set.scaler_y))
        #test_loader = self._get_data(flag='test')

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        #path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if test:
            print('loading model')
            best_model_path = path + '/' + 'best_model.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                 #Plotting1
                if i % 5 ==0:
                    fig, ax = plt.subplots(2,1,figsize=[12, 8])
                    ax[0].plot(true[0].reshape(-1, 1)[:6800])
                    ax[0].plot(pred[0].reshape(-1, 1)[:6800])
                    ax[1].plot(true[0].reshape(-1, 1)[-6800:])
                    ax[1].plot(pred[0].reshape(-1, 1)[-6800:])
                    # plt.legend(['Orig', 'Pred'])
                    # plt.title('Compare prediction')
                    plt.savefig(os.path.join(folder_path, f'compare_prediction_test0_{i}.pdf'),bbox_inches='tight')
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, pred=False):
        train_set, train_loader = self._get_data(flag='train')
        pred_set,pred_loader = self._get_data(flag='pred', scalers=(train_set.scaler_x,train_set.scaler_y))

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        if not os.path.exists(path):
            os.makedirs(path)

        if pred:
            print('loading model')
            best_model_path = path + '/' + 'best_model.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                # result save
                folder_path = './pred_results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                ## plotting
                fig, ax = plt.subplots(2,1,figsize=[12, 8])
                ax[0].plot(true.reshape(-1, 1)[:6800//3])
                ax[0].plot(pred.reshape(-1, 1)[:6800//3])
                ax[1].plot(true.reshape(-1, 1)[-6800//3:])
                ax[1].plot(pred.reshape(-1, 1)[-6800//3:])
                # plt.legend(['Orig', 'Pred'])
                # plt.title('Compare prediction')
                plt.savefig(os.path.join(folder_path, 'compare_prediction_pred.pdf'), bbox_inches='tight')
                print('finishing')
        return

