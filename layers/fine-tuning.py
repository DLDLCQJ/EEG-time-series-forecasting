import torch
from torch import nn
import torch.nn.functional as F


class Attnfinetuning(nn.Module):
    def __init__(self, input_dim,h_dim,pred,d_model,  dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.pred = pred
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_model*4, out_channels=d_model, kernel_size=1)
        self.activation = F.relu
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, input_dim,bias=True)

    def forward(self, g):  #[bs, pred, 1]
        
        y = self.dropout(self.activation(self.conv1(g.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        out = self.norm(g+y)
        ##
        out = self.fc(out)
        return out
