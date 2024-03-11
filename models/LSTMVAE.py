
import torch
from torch import nn, 
from torch.nn import functional as F

class Creating_h0(nn.Module):
    def __init__(self):
        super(Creating_h0,self).__init__()

        self.fc1 = nn.Linear(6801,64*5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64*5,64*2)
        
        #self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        
        h0 = self.fc2(self.relu(self.fc1(x)))
    
        return h0

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim,latent_dim,num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.rnn = nn.LSTM(input_dim, hidden_dim,num_layers,
                           batch_first=True, bidirectional=False,)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        
    def reparametize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) 

        z = mu + eps * std
        return z

    def forward(self, x, h0):

        # x: tensor of shape (batch_size, seq_length, input_dim)
        batch_size, seq_len, feature_size = x.shape
        output, (hidden, cell) = self.rnn(x,(h0.detach(),h0.detach()))
        
        enc_h = hidden.view(batch_size, self.hidden_dim)
        # extract latent variable z(hidden space to latent space)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mean = self.fc21(enc_h) #[bs,latent_dim]
        log_var = self.fc22(enc_h)
        ###################
        z0 = self.reparametize(mean, log_var)  # batch_size x latent_dim
        
        return (z0, mean, log_var,(hidden, cell)) #[bs, latent_size]

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,num_layers=2):
        super().__init__()
        """
        input_dim: int, batch_size x sequence_length x input_dim
        hidden_dim: int, output size of LSTM AE
        latent_dim: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        #self.dropout = dropout
        self.num_layers = num_layers
        self.rnn = nn.LSTM(latent_dim, hidden_dim,num_layers,
                           batch_first=True, bidirectional=False,)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden): #

        output, (hidden, cell) = self.rnn(x, hidden)
        #output = F.dropout(output, p=self.dropout)
        #?????????????
        prediction = self.fc(output)

        return prediction, (hidden, cell)
    
class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim,latent_dim,num_layers=2):
        super(LSTMVAE, self).__init__()
    
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        ##
        self.encoder = Encoder(input_dim, hidden_dim,latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim,input_dim, num_layers)
        
    def forward(self, x, h0): #x: bs,input_size
        batch_size, seq_len, feature_size = x.shape

        # encode input space to hidden space
        z0_enc_hidden = self.encoder(x,h0) #(mu,log_var); enc_hidden[0/1]:torch.Size([num_layers, bs, hidden_dim])
        
        ####################
        z0 = z0_enc_hidden[0]
        mean = z0_enc_hidden[1]
        log_var = z0_enc_hidden[2]
        enc_hidden = z0_enc_hidden[3]
        # decode latent space to input space
        z = z0.repeat(1, seq_len, 1) ##??????
        z = z.view(batch_size, seq_len, self.latent_dim)
        
        #####################
        reconstruct_output, hidden = self.decoder(z,  enc_hidden)

        x_hat = reconstruct_output
        
        
        return (x_hat, z0, mean, log_var)
