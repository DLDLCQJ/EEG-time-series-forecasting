
import torch
from torch import nn, 
from torch.nn import functional as F

class AttnGenerator(nn.Module):
    def __init__(self, input_dim,h_dim, d_model,pred, head,rank, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.pred = pred
        #################
        self.dropout = nn.Dropout(dropout)
        self.ps1 = nn.PixelShuffle(5)
        self.ps2 = nn.PixelShuffle(5)
        self.ps3 = nn.PixelShuffle(4)
        self.ps4 = nn.PixelShuffle(4)
        self.gan_embedding = PositionalEmbedding(d_model,pred+h_dim)
        self.block1 = Block_Self(d_model, pred+h_dim,head, rank, dropout)
        self.block2 = Block_Cross(d_model, h_dim, head, rank, dropout)
        self.block2 = Block_Self(25*latent_dim, head, rank, dropout)
        self.block3 = Block_Self(64*latent_dim, head, rank, dropout)
        self.block4 = Block_Self(256*latent_dim, head, rank, dropout)
 
    def forward(self, x):  #[1, 32, 128]
        #x = x.permute(1,2,0)
        bs = x.size(0)
        ##
        g_int = torch.zeros([bs, self.pred, self.input_dim]).float().to(device)
        g_e = self.gan_embedding(g_int)
        g_cat0 = torch.cat([x,g_e], dim =1)
        #
        g = self.block1(g_cat0)
        #
        out = self.block2(g, x, x)
        ##
       
        out2 = self.ps2(out1) #[bs, -1, 16, 16*latn]
        print(out2.shape)
        out2 = out2.view(-1,25,25*self.latent_dim)
        out2 = self.block2(out2)

        #
        out3 = self.ps3(out2)
        out3 = self.block3(out3)
        ##
        out4 = self.ps4(out3)
        out4 = self.block3(out4)

        return out4[:,-self.pred:,:]
