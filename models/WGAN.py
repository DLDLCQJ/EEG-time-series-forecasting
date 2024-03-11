import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru_1 = nn.GRU(1, 128, batch_first=True, num_layers=2)
        self.gru_2 = nn.GRU(128, 256, batch_first=True)
        self.gru_3 = nn.GRU(256, 512, batch_first=True)
        self.linear_1 = nn.Linear(512, 1024); self.linear_2 = nn.Linear(1024, 2048); self.linear_3 = nn.Linear(2048, 2267)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out_1, _ = self.gru_1(x, torch.zeros(1, x.size(0), 128).to(device))
        out_2, _ = self.gru_2(self.dropout(out_1), torch.zeros(1, x.size(0), 256).to(device))
        out_3, _ = self.gru_3(self.dropout(out_2), torch.zeros(1, x.size(0), 512).to(device))
        out_4 = self.linear_1(self.dropout(out_3)[:, -1, :]);  out_5 = self.linear_2(out_4); out_6 = self.linear_3(out_5)
        return out_6


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(9068, 9068//2, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv1d(4534, 4534//2, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv1d(2267, 128, kernel_size=5, stride=1, padding='same')
        self.linear1 = nn.Linear(128, 64); self.linear2 = nn.Linear(64, 32); self.linear3 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.01); self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.leaky_relu(self.conv1(x)); conv2 = self.leaky_relu(self.conv2(conv1)); conv3 = self.leaky_relu(self.conv3(conv2))
        out_1 = self.leaky_relu(self.linear1(conv3.view(conv3.size(0), -1))); out_2 = self.leaky_relu(self.linear2(out_1))
        return self.sigmoid(self.linear3(out_2))
