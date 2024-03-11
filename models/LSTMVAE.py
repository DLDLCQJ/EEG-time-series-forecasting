
import torch
from torch import nn, 
from torch.nn import functional as F

class LSTMVAE(nn.Module):
    def __init__(self, latent_dim=64, num_layers=1):
        super(LSTMVAE, self).__init__()
        self.fc_hidden_1 = nn.Linear(10000, 2 * latent_dim)
        self.fc_hidden_2 = nn.Linear(2 * latent_dim, latent_dim)
        self.fc_encoder = nn.Linear(1, latent_dim)
        self.rnn_encoder = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(2 * latent_dim, 2 * latent_dim)
        self.fc_logvar = nn.Linear(2 * latent_dim, 2 * latent_dim)
        self.rnn_decoder = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc_decoder = nn.Linear(latent_dim, 1)

    def reparametize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)
        
    def forward(self, x, h):
        h = self.fc_hidden_1(h)
        h = self.fc_hidden_2(h)
        output = self.fc_encoder(x)
        output, (hidden_state, cell_state) = self.rnn_encoder(output, (h.unsqueeze(0), h.unsqueeze(0)))
        mean = self.fc_mu(torch.concat((hidden_state, cell_state), dim=1))
        logvar = self.fc_logvar(torch.concat((hidden_state, cell_state), dim=1))
        z = self.reparametize(mean, logvar)
        output, (hidden_state, cell_state) = self.rnn(z.repeat(1, x.shape[1], 1), (z, z))
        output = self.fc_decoder(output)
        return (output, mean, logvar, z)
    
    def encoder(self, x, h):
        h = self.fc_hidden_1(h)
        h = self.fc_hidden_2(h)
        output = self.fc_encoder(x)
        output, (hidden_state, cell_state) = self.rnn_encoder(output, (h.unsqueeze(0), h.unsqueeze(0)))
        mean = self.fc_mu(torch.concat((hidden_state, cell_state), dim=1))
        logvar = self.fc_logvar(torch.concat((hidden_state, cell_state), dim=1))
        return (mean, logvar)
