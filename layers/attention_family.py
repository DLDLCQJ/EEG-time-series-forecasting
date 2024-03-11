import torch
from torch import nn, optim
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    '''Position embedding'''
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float().to(device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)


    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return x
     
class MultiHead_SelfAttention(nn.Module):
    def __init__(self, attn_dim, len_dim, head,  rank,  dropout=0.1):
        super().__init__()
        self.head = head
        self.attn_dim = attn_dim
        self.a_h = attn_dim // head
        self.l_r = len_dim // rank
        self.rank = rank
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(attn_dim, attn_dim)
        self.linear_key = nn.Linear(attn_dim, attn_dim)
        self.linear_value = nn.Linear(attn_dim, attn_dim)
        ####?#####
        self.convs = nn.Conv1d(rank*rank, len_dim,kernel_size=1)
        self.factor = nn.Parameter(torch.Tensor(1,  self.a_h, self.l_r, rank))
        nn.init.xavier_normal_(self.factor)
    def forward(self, x, mask=None):
        bs = x.size(0)
        '''
        x:
        self.linear_query(x):[1,8,a_h,l_r,head*rank]

        factor:[1,8,a_h,head*rank,l_r]

        '''
        query = self.linear_query(x).view(bs, self.a_h, self.head*self.rank ,self.l_r)
        key = torch.matmul(self.linear_key(x).view(bs, self.a_h, self.head*self.rank ,self.l_r),self.factor)
        value = torch.matmul(self.linear_value(x).view(bs, self.a_h, self.head*self.rank ,self.l_r),self.factor)

        ##
        '''
        query:[1, 8, a_h,l_r,l_r]
        scores:[1,8,-1, -1]
        value:[1, 8, -1,rank]
        attn:[1,8,-1,rank]
        '''
        scores = torch.matmul(query.transpose(-2, -1), key) / math.sqrt(self.head*self.rank)
        self_attn = F.softmax(scores, dim=-1)
        self_attn = torch.matmul(self_attn, value.transpose(-2, -1))
        return self_attn.view(bs, -1,self.attn_dim)

class MultiHead_CrossAttention(nn.Module):
    def __init__(self, attn_dim, K_dim, head, rank, dropout=0.1):
        super().__init__()
        self.head = head
        self.attn_dim = attn_dim
        self.a_h = attn_dim // head
        self.k_r = K_dim // rank
        self.rank = rank
        self.dropout = nn.Dropout(dropout)
        self.linear_Q = nn.Linear(attn_dim, attn_dim)
        self.linear_K = nn.Linear(attn_dim, attn_dim)
        self.linear_V = nn.Linear(attn_dim, attn_dim)
        self.factor = nn.Parameter(torch.Tensor(1,  self.a_h, self.k_r, rank))
        nn.init.xavier_normal_(self.factor)
    def forward(self, Q, K, V, mask=None):
  
        bs = Q.size(0)
        Q = self.linear_Q(Q).view(bs, self.a_h, self.head*self.rank , -1)
        K = torch.matmul(self.linear_K(K).view(bs, self.a_h, self.head*self.rank , self.k_r),self.factor)
        V = torch.matmul(self.linear_V(V).view(bs, self.a_h, self.head*self.rank , self.k_r),self.factor)
        ##  
        scores = torch.matmul(Q.transpose(-2, -1), K) / math.sqrt(self.head*self.rank)
        cross_attn = F.softmax(scores, dim=-1)
        cross_attn =  torch.matmul(cross_attn, V.transpose(-2, -1))
        return cross_attn.view(bs, -1,self.attn_dim)

class FeedForwardBlock(nn.Sequential):
    def __init__(self, input_size, expansion, drop):
        super().__init__(
            nn.Linear(input_size, expansion * input_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(expansion * input_size, input_size),
        )

class Block_Self(nn.Module):
    def __init__(self, attn_dim, len_dim, head,rank, dropout=0.1):
        super().__init__()
        self.attn = MultiHead_SelfAttention(attn_dim,len_dim, head, rank)
        self.FeedForwardBlock = FeedForwardBlock(attn_dim, 2, dropout)
        self.norm = nn.LayerNorm(attn_dim)
    def forward(self, x):
        attn_out = self.FeedForwardBlock(self.attn(x))
        out = self.norm(attn_out+x)
        return out

class Block_Cross(nn.Module):
    def __init__(self, attn_dim, k_dim, head,rank, dropout=0.1):
        super().__init__()
        self.attn = MultiHead_CrossAttention(attn_dim, k_dim, head, rank)
        self.FeedForwardBlock = FeedForwardBlock(attn_dim, 2, dropout)
        self.norm = nn.LayerNorm(attn_dim)
    def forward(self, Q,K,V):
        attn_out = self.FeedForwardBlock(self.attn(Q,K,V))
        out = self.norm(attn_out+Q)
        return out
