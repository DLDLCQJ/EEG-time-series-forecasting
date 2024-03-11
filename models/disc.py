import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

class AttnDiscriminator(nn.Module):
    def __init__(self, input_dim, tw_st,head,rank, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.tw_st = tw_st
        self.vp1 = nn.AvgPool2d(5)
        self.vp2 = nn.AvgPool2d(5)
        self.vp3 = nn.AvgPool2d(4)
        self.vp4 = nn.AvgPool2d(4)
        self.block1 = Block_Self(25,tw_st,head, rank, dropout)
        self.block2 = Block_Self(5*latent_dim, head, rank, dropout)
        self.block3 = Block_Self(4*latent_dim, head, rank, dropout)
        self.block4 = Block_Self(256*latent_dim, head, rank, dropout)
    def forward(self, x):  #[bs, pred+tw, 1]
        B, L, F = x.size()
        x = x.view(B,-1, 25, 25)
        ##
        out1 = self.vp1(x)
        out1 = self.block1(out1)
        ##
        out2 = self.vp2(out1)
        out2 = self.block2(out2)
        ##
        out3 = self.vp3(out2)
        out3 = self.block3(out3)
        ##
        out4 = self.v4(out3)
        out4 = self.block4(out4)

        return out4.view(B,-1,self.input_dim)
