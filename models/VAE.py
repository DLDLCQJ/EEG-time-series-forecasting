import torch
from torch import nn, optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim,d_model,num_layers,tw, head,rank):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        ##
        self.enc_embedding = PositionalEmbedding(d_model,tw)
        #self.rnn_encoder = nn.LSTM(d_model, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.blocks = Block_Self(d_model, tw, head, rank)
        self.convs = ConvLayer(d_model)
        #self.fc = nn.Linear(latent_dim,  input_dim)
        # ##
        self.fc_mu_latn = nn.Linear(latent_dim,  latent_dim)
        self.fc_logvar_latn = nn.Linear(latent_dim, latent_dim)
        self.fc_mu_attn = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar_attn = nn.Linear(latent_dim, latent_dim)
    def forward(self, x, mask=None):
        '''
        out depend on two factors: latn and attn
        '''
        x_e = self.enc_embedding(x)
        out, (h, c) = self.rnn_encoder(x_e, (h,h)) #out:[bs,len,latn]##
        ##latn
        mean_latn = self.fc_mu_latn(c) 
        logvar_latn = self.fc_logvar_latn(c)
        #attn
        out_attn = self.blocks(x_e) #out_attn:[ bs,len, latn]
        ###########dim_down############
        out_attn_c = self.convs(out_attn)
        out_attn_c = self.fc(out_attn_c)
        #####################################
        mean_attn = self.fc_mu_attn(out_attn_c) 
        logvar_attn = self.fc_logvar_attn(out_attn_c) 
        print(mean_latn.shape,mean_attn.shape)
        return out_attn_c

##
class Decoder_ATTN(nn.Module):
    def __init__(self, input_dim, h_dim,d_model,num_layers,tw,head,rank):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.tw = tw
        ##recon_attn
        #self.rnn_decoder_attn = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.blocks = Block_Self(d_model, h_dim, head, rank)
        self.convs = nn.Conv1d(h_dim,tw,kernel_size=1)
        self.fc_attn =  nn.Sequential(nn.Linear(d_model, input_dim), nn.Sigmoid())
       
    def reparametize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)
    def forward(self, x, mask=None): 
        bs = x.size(0)
        #recon_attn
        z_attn = self.reparametize(mu, logvar)#[bs,h_dim,latn]
        out, (h, c) = self.rnn_decoder_attn(x, hidden)
        out_attn = self.blocks(x) #[bs,h_dim,latn]
        ###########dim_up############
        out_attn_c = self.convs(out_attn) #[bs,len,latn]
        recon_out_attn = self.fc_attn(out_attn_c)
        return recon_out_attn
