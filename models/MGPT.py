import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, DataEmbedding,MusEmbedding
from models.Encoder import Encoder,ConvLayer
from models.Generator import decoder
from layers.Convfamily import UNetBlock, ConvMaxpool, Attn_Gate_Update2,FeedForwardAdapter,Up_Mome,Down_Mome
from layers.Attenfamily import Full_Attn,LoRa_Attn,Prob_Attn, AttentionLayer,Attn_Blocks
from models.Reparametize import Reparametize
from models.Finetuning import Attnfinetuning

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.enc_embedding = MusEmbedding(configs.enc_in,configs.d_model,configs.seq_len)
        self.encoder = Encoder(
                    [
                    Attn_Blocks(
                    AttentionLayer(
                    Full_Attn(False,scale = True,dropout=configs.dropout)
                    ,configs.d_model, configs.n_heads)
                    ,configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout
                    )
                    # Attn_Blocks(
                    # AttentionLayer(
                    # Prob_Attn(False, configs.factor, dropout=configs.dropout)
                    #         ,configs.d_model, configs.n_heads)
                    #         ,configs.d_model,
                    #         configs.d_ff,
                    #         dropout=configs.dropout
                    #     ) 
                    for l in range(configs.e_layers)
                    ]
                    ,[
                    ConvLayer(
                        configs.d_model
                        ) for l in range(configs.e_layers - 1)
                    ] if configs.distil else None,
                    norm_layer=nn.LayerNorm(configs.d_model)
                )

        # decoder
        
                        # decoder( 
                        #     Down_Mome(
                        #         [UNetBlock(
                        #         configs.seq_len//configs.stride_d**l,
                        #         configs.stride_d, configs.scale_factor,configs.d_model, down=True, gene=False, dropout=configs.dropout
                        #          ) for l in range(configs.d_layers)],

                        #         [nn.GRU(
                        #         input_size = 1 if l == configs.u_layers-1 else configs.d_model//2,
                        #         hidden_size = configs.d_model//2,
                        #         num_layers= configs.num_layers,
                        #         batch_first=True,
                        #         ) for l in range(configs.d_layers)],
                        #         configs.d_model
                        #         ),
                        
        self.decoder =  decoder(
                            Up_Mome(
                            nn.Sequential(nn.Linear(configs.label_len, 5 * (configs.d_model//2)),
                                          nn.ReLU(),
                                          nn.Linear(5 * (configs.d_model//2),  configs.d_model//2)),

                            [nn.GRU(
                            input_size = 1 if l == configs.g_layers-1 else configs.d_model//2,
                            hidden_size = configs.d_model//2,
                            num_layers= configs.num_layers,
                            batch_first=True,
                            ) for l in range(configs.g_layers)],

                            nn.Sequential(nn.Linear(
                            configs.d_model//2,
                            configs.pred_len,
                            #configs.c_out,
                            bias=True),
                            #nn.Sigmoid()
                            #configs.seq_len//configs.u_layers,bias=True
                            ),
                            # for l in range(configs.g_layers)],
                            configs.num_layers,
                            configs.d_model
                            ),

                            [UNetBlock(
                            in_channels = configs.label_len*(configs.stride_g**l),
                            stride_g = configs.stride_g,scale_factor=configs.scale_factor,d_model=configs.d_model,down=False,gene=True,dropout=configs.dropout
                            ) for l in range(configs.g_layers)],

                            [MusEmbedding(configs.c_out,configs.d_model,configs.pred_len) for l in range(configs.g_layers)],

                            [Attn_Gate_Update2(
                                #Attn_Blocks(
                                AttentionLayer(
                                Prob_Attn(False, configs.factor, dropout=configs.dropout)
                                ,configs.d_model, configs.n_heads
                                ),
                                # configs.d_model,
                                # configs.d_ff,
                                # dropout=configs.dropout
                                # ),
                                # Attn_Blocks(
                                # AttentionLayer(
                                # Prob_Attn(False, configs.factor, dropout=configs.dropout)
                                # ,configs.d_model, configs.n_heads
                                # ),  
                                # configs.d_model,
                                # configs.d_ff,
                                # dropout=configs.dropout
                                # )
                                Attn_Blocks(
                                AttentionLayer(
                                Full_Attn(False,scale = True,dropout=configs.dropout)
                                ,configs.d_model, configs.n_heads
                                ),
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout
                                ),
                                # Attn_Blocks(
                                # AttentionLayer(
                                # Full_Attn(False,scale = True,dropout=configs.dropout)
                                # ,configs.d_model, configs.n_heads
                                # )
                                # ,configs.d_model,
                                # configs.d_ff,
                                # dropout=configs.dropout
                                # ),
                                #(configs.pred_len//(configs.stride_g ** (configs.g_layers-l-1))),configs.d_model
                                ) for l in range(configs.g_layers)],

                            [nn.Linear(configs.d_model, configs.c_out, bias=True) for l in range(configs.g_layers)],   
                            configs.d_model              
                            )    

    def forward(self, x_enc,stoken):
        enc_em = self.enc_embedding(x_enc)
        cross = self.encoder(enc_em)
        z = self.reparametize(enc_out) 
        stoken = enc_em[:, -self.label_len:, :]
        dec_em = self.dec_embedding(stoken)
        dec_out = self.decoder(x_enc,stoken)
        out = self.finetuning(dec_out[:,-self.pred_len:,:])
        return out
    
