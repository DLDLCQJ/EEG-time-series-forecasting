import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

class LSTMGenerator(nn.Module):

    def __init__(self, in_dim, out_dim=2267, n_layers=1, hidden_dim=128):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        ##self.lstm(input) input:[bs, seq_len, 1]
        self.lstm1 = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_dim, 256, n_layers, batch_first=True)
        # self.lstm3 = nn.LSTM(256, 512, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(128, out_dim), nn.Sigmoid())
        # self.linear = nn.Sequential(nn.Linear(hidden_dim, 1024), 
        #                             nn.ReLU(),
        #                             nn.Dropout(0.2),
        #                             nn.Linear(1024, 2048),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.2),
        #                             nn.Linear(2048, out_dim),
        #                             nn.Sigmoid())
    def forward(self, input):
        batch_size = input.size(0)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        recurrent_feature1, _ = self.lstm1(input, (h_0, c_0)) #torch.Size([60, 64, 128])
        # h_1 = torch.zeros(self.n_layers, batch_size, 256).to(device)
        # c_1 = torch.zeros(self.n_layers, batch_size, 256).to(device)
        # recurrent_feature2, _ = self.lstm2(recurrent_feature1, (h_1, c_1)) #torch.Size([60, 64, 128])
        # h_2 = torch.zeros(self.n_layers, batch_size, 512).to(device)
        # c_2 = torch.zeros(self.n_layers, batch_size, 512).to(device)
        # recurrent_feature3, _ = self.lstm3(recurrent_feature2, (h_2, c_2)) #torch.Size([60, 64, 128])
        #print(recurrent_features.shape)
        #enc_h = hidden.view(batch_size, self.hidden_dim)
        outputs = self.linear(recurrent_feature1[:, -1, :])
        
        return outputs
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

    def  forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
          self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        #If channel_last, the expected format is (batch_size, seq_len, features)
        y1 = self.tcn(x.transpose(1, 2))
        return self.linear(y1.transpose(1, 2))

class CausalConvGenerator(nn.Module):

    def __init__(self, noise_size=1, output_size=2267, n_layers=8, n_channel=10, kernel_size=8, dropout=0):
        super().__init__()
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(noise_size, output_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(64, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(self.tcn(x).transpose(1, 2)))

class CausalConvDiscriminator(nn.Module):
    def __init__(self, input_size=9068, n_layers=8, n_channel=10, kernel_size=8, dropout=0):
        super().__init__()
        #Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, num_channels, kernel_size, dropout)
        
    def forward(self, x):
        return self.tcn(x.transpose(1, 2))
        #return self.tcn(x, channel_last)
    
 
