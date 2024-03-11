import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Disc_Unet import discinimator,UNet_Down_Disc, UNet_Up_Disc
from layers.Embed import PositionalEmbedding, DataEmbedding,EEGEmbedding
from models.Encoder import Encoder,ConvLayer
from models.Generator import decoder#, Decoder, DecoderLayer
from layers.Convfamily import UNetBlock, ConvMaxpool, Attn_Gate_Update1,Attn_Gate_Update2,FeedForwardAdapter,ResBlock_Disc
from layers.Attenfamily import Full_Attn,LoRa_Attn,Prob_Attn, AttentionLayer,Attn_Blocks
from models.Reparametize import Reparametize
from models.Finetuning import Attnfinetuning

class gene(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        # Embedding
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.enc_embedding = PositionalEmbedding(configs.d_model, configs.seq_len)
        # self.dec_embedding = PositionalEmbedding(configs.d_model, configs.pred_len+configs.label_len)
        self.enc_embedding = EEGEmbedding(configs.enc_in,configs.d_model,configs.seq_len)
        self.stoken_embedding = EEGEmbedding(configs.dec_in, configs.d_model,configs.label_len)
        #self.dec_embedding = EEGEmbedding(configs.dec_in, configs.d_model,configs.pred_len)

        # Encoder
        self.encoder = Encoder(
            [
            Attn_Blocks(
                AttentionLayer(
                Prob_Attn(False, configs.factor, dropout=configs.dropout
                          )
                        ,configs.d_model, configs.n_heads
                        )
                        ,configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout
                   ) 
                for l in range(configs.e_layers)
            ]
            ,[
            ConvLayer(
                configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        ## decoder
        self.decoder = decoder(
                                [EEGEmbedding(configs.d_model, configs.d_model,configs.label_len*(configs.stride_g**l)) for l in range(configs.g_layers)],
                                [Attn_Gate_Update1(
                                Attn_Blocks(
                                AttentionLayer(
                                Prob_Attn(False, configs.factor, dropout=configs.dropout
                                        )
                                        ,configs.d_model, configs.n_heads)   
                                        ,configs.d_model,
                                        configs.d_ff,
                                        dropout=configs.dropout
                                        )
                                        ,configs.seq_len//configs.stride_e**(configs.e_layers-1),configs.d_model
                                        )for l in range(configs.g_layers)],
                                                   
                                [UNetBlock(configs.pred_len//(configs.stride_g ** (configs.g_layers-l)), 
                                           configs.stride_g, configs.scale_factor, configs.d_model,
                                           down=False, gene=True,dropout=configs.dropout
                                           ) for l in range(configs.g_layers)],

                                [EEGEmbedding(configs.d_model, configs.d_model,configs.label_len*(configs.stride_g**(l+1))) for l in range(configs.g_layers)],
                                              
                                [Attn_Gate_Update2(

                                Attn_Blocks(
                                AttentionLayer(
                                Prob_Attn(False, configs.factor, dropout=configs.dropout)
                                ,configs.d_model, configs.n_heads
                                ),
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout
                                )
                                ,Attn_Blocks( 
                                AttentionLayer(
                                Prob_Attn(False, configs.factor, dropout=configs.dropout)
                                ,configs.d_model, configs.n_heads
                                ),  
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout
                                ), 

                                configs.pred_len//(configs.stride_g ** (configs.g_layers-l-1)),configs.d_model
                                ) for l in range(configs.g_layers)],

                                nn.Sequential(nn.Linear(configs.d_model, configs.c_out, bias=True)),                  
                                )
  
    def forward(self, x_enc,stoken):
        enc_em = self.enc_embedding(x_enc)
        #pos_em = self.dec_embedding(pos_pre)
        cross, scores = self.encoder(enc_em)
        #z = self.reparametize(enc_out) 
        
        #stoken = enc_em[:, -self.label_len:, :]
        dec_out = self.decoder(stoken,cross)
        #dec_out =dec_out+pos_em
        return dec_out 
    
    

class disc(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        # Embedding
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.enc_embedding = PositionalEmbedding(configs.d_model, configs.seq_len)
        # self.dec_embedding = PositionalEmbedding(configs.d_model, configs.pred_len+configs.label_len)
        self.Disc_embedding = EEGEmbedding(configs.dec_in, configs.d_model)
        ## Disc
        #down
        Down_Blocks = nn.ModuleList()
        Median_Block = nn.ModuleList()
        #en_channels = configs.seq_len//(configs.stride_e ** (configs.e_layers-1))
        for i in range(configs.d_layers):
            #out_channels =  (configs.pred_len+configs.label_len) if i ==0 else (configs.pred_len+configs.label_len)//(configs.stride_d ** i)
            out_channels = (configs.pred_len) if i ==0 else (configs.pred_len)//(configs.stride_d ** i)
            ##    
            if i == configs.g_layers -1:
                res_block = ResBlock_Disc(out_channels, out_channels,configs.d_model)
                # cross_attn_adapter = Attn_Blocks(Multihead_CrossAttention,
                #                             out_channels, configs.label,
                #                             configs.d_model, configs.n_heads,configs.win,
                #                             dropout=configs.dropout
                #                             )                           
                unet_down_block = UNetBlock(out_channels, configs.stride_d, configs.scale_factor,configs.d_model, down=True, gene=False, dropout=configs.dropout)
                
                Down_block_ls =  UNet_Down_Disc(unet_down_block=unet_down_block,res_block=res_block) 
                Down_Blocks.append(Down_block_ls) 
                ##
                m_channels = out_channels//configs.stride_d
                #res_block = ResBlock(m_channels, configs.label_len)
                # cross_attn_adapter = Attn_Blocks(Multihead_CrossAttention,
                #                             m_channels,configs.label,
                #                             configs.d_model, configs.n_heads,configs.win,
                #                             dropout=configs.dropout
                #                             ) 
                ##
                res_block_median = ResBlock_Disc(m_channels,m_channels,configs.d_model)
                Median_block =  UNet_Down_Disc(median_block=True,res_block_median=res_block_median) 
                Median_Block.append(Median_block)  
            else:
                res_block = ResBlock_Disc(out_channels, out_channels,configs.d_model)
                # cross_attn_adapter =  Attn_Blocks(Multihead_CrossAttention,
                #                             out_channels,configs.label,
                #                             configs.d_model, configs.n_heads,configs.win,
                #                             dropout=configs.dropout
                #                             ) 
                unet_down_block = UNetBlock(out_channels, configs.stride_d, configs.scale_factor, configs.d_model,down=True, gene=False, dropout=configs.dropout)     
                Down_block = UNet_Down_Disc(unet_down_block=unet_down_block,res_block=res_block)
                Down_Blocks.append(Down_block)  
        #up
        Up_Blocks = nn.ModuleList()
        for i in reversed(range(configs.d_layers)):
            ##
            in_channels = (configs.pred_len)//configs.stride_d if i ==0 else (configs.pred_len)//(configs.stride_d ** (i+1))
            out_channels = in_channels*configs.stride_d
            unet_up_block = UNetBlock(in_channels, configs.stride_d, configs.scale_factor,configs.d_model, down=False, gene=False,dropout=configs.dropout)
            res_block = ResBlock_Disc(out_channels, out_channels,configs.d_model)
            # cross_attn_adapter = Attn_Blocks(Multihead_CrossAttention,
            #                             out_channels,configs.label,
            #                             configs.d_model, configs.n_heads,configs.win,
            #                             dropout=configs.dropout
            #                             ) 
            Up_block = UNet_Up_Disc(out_channels, unet_up_block=unet_up_block,res_block=res_block) 
            Up_Blocks.append(Up_block)              
        ##      
        self.discriminator = discinimator(Down_Blocks,Median_Block,Up_Blocks,configs.pred_len)
        self.projecting_D = nn.Linear(configs.d_model, configs.input_dim)
        self.projecting_M = nn.Linear(configs.d_model, configs.input_dim)
    def forward(self,x):
        disc_em = self.Disc_embedding(x)
        D_outs, M_outs = self.discriminator(disc_em)
        D_out = self.projecting_D(D_outs[-1])
        M_out = self.projecting_M(M_outs[-1])
        return D_out, M_out
    
# class gene(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.dec_embedding = EEGEmbedding(configs.dec_in,configs.d_model)

#         ## decoder
#         self.decoder = decoder(
#                                 [Attn_Gate_Update1(
#                                 #Attn_Blocks(
#                                 AttentionLayer(
#                                 Prob_Attn(False, configs.factor, dropout=configs.dropout
#                                         )
#                                         ,configs.d_model, configs.n_heads)   
#                                         # ,configs.d_model,
#                                         # configs.d_ff,
#                                         # dropout=configs.dropout
#                                         # )
#                                         ,configs.seq_len//configs.stride_e**(configs.e_layers-1),configs.d_model
#                                         )for l in range(configs.g_layers)],
                                                   
#                                 [UNetBlock(configs.pred_len//(configs.stride_g ** (configs.g_layers-l)), 
#                                            configs.stride_g, configs.scale_factor, configs.d_model,
#                                            down=False, gene=True,dropout=configs.dropout
#                                            ) for l in range(configs.g_layers)],

#                                 [Attn_Gate_Update2(

#                                 # Attn_Blocks(
#                                 AttentionLayer(
#                                 Prob_Attn(False, configs.factor, dropout=configs.dropout)
#                                 ,configs.d_model, configs.n_heads
#                                 ),
#                                 # ,configs.d_model,
#                                 # configs.d_ff,
#                                 # dropout=configs.dropout
#                                 # )
#                                 #,Attn_Blocks(
#                                 AttentionLayer(
#                                 Prob_Attn(False, configs.factor, dropout=configs.dropout)
#                                 ,configs.d_model, configs.n_heads
#                                 ),  
#                                 # ,configs.d_model,
#                                 # configs.d_ff,
#                                 # dropout=configs.dropout
#                                 # ), 

#                                 (configs.pred_len//(configs.stride_g ** (configs.g_layers-l-1))),configs.d_model
#                                 ) for l in range(configs.g_layers)],

#                                 [nn.Linear(configs.d_model, configs.c_out, bias=True)for l in range(configs.g_layers)],                  
#                                 )
#     def forward(self,x,cross):
#         dec_em = self.dec_embedding(x)
#         #z = self.reparametize(x)
#         dec_out = self.decoder(dec_em,cross)
       
#         return dec_out
