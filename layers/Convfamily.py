import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ConvMaxpool(nn.Module):
    def __init__(self, c_in, stride,d_model):
        super().__init__()
        self.Conv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
    def forward(self, x):
        x = self.Conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x.permute(0, 2, 1))
        x = x.transpose(1,2)
        return x

class InterpolateUpsample(nn.Module):
    def __init__(self, c_in, scale_factor,d_model):
        super().__init__()
        self.scale_factor = scale_factor
        self.Conv = nn.Conv1d(c_in, 
                                c_in, 
                                kernel_size=3, 
                                padding=1,
                                padding_mode='circular')
        #self.norm = nn.LayerNorm(d_model)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.Conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = F.interpolate(x.permute(0, 2, 1), scale_factor=self.scale_factor, mode='linear', align_corners=False)
        x = x.transpose(1,2)
        return x

class ResBlock_Disc(nn.Module):
    def __init__(self, channels1,channels2,d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(channels1, channels2, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels2, channels1, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        ##
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        ##
        out = self.conv2(out)
        out = self.norm2(out)
        out += x  # Skip connection
        out = self.relu(out)  # Another ReLU after adding the residual
        return out
    
class ResBlock_Gene(nn.Module):
    def __init__(self, in_channels, out_channels,d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model) 
    def forward(self,x, cross=None):
        residual =x
        ##
        out = self.conv1(x)
        out = self.norm1(out+cross)
        out = self.relu(out)
        ##
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.relu(out)
        return out

class Gate_Update(nn.Module):
    def __init__(self, in_channels, out_channels,d_model):
        super().__init__()
        self.out_channels = out_channels
        self.activate = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels+out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.candidate_state = nn.Sequential(
            nn.Conv1d(in_channels+out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh())
    def forward(self, x1, x2):
        combine = torch.cat([x1, x2], dim=1)
        combine = self.norm1(self.conv(combine))
        gates = self.activate(combine)
        reset_comb = torch.mul(gates, combine)
        candidate_input = torch.cat([reset_comb, x1], dim=1)
        candidate_state = self.candidate_state(candidate_input)
        ##
        x2_state = (1 - gates) * candidate_state + torch.mul(gates,x2)
        x2_state = self.norm2(x2_state)
        x2_state += x2
        x2_state = self.activate(x2_state)
        return x2_state
    
class Gate_Update1(nn.Module):
    def __init__(self, in_channels, out_channels,d_model,dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        #self.norm = nn.LayerNorm(d_model)
        self.activate = nn.Sigmoid()
        self.gates = nn.Sequential(
            nn.Conv1d(in_channels+out_channels, 2*out_channels, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.dropout_reset = nn.Dropout(dropout)
        self.dropout_update = nn.Dropout(dropout)
        self.candidate_state = nn.Sequential(
            nn.Conv1d(in_channels+out_channels, out_channels,kernel_size=3, padding=1),
            nn.Tanh())
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gates = self.gates(combined)
        reset_gate, update_gate = self.dropout_reset(torch.sigmoid(gates[:, :self.out_channels,:])), self.dropout_update(torch.sigmoid(gates[:, self.out_channels:,:]))
        reset_x2 = torch.mul(reset_gate, x2)
        candidate_input = torch.cat([x1,reset_x2], dim=1)
        candidate_state = self.candidate_state(candidate_input)
        x2_state = (1 - update_gate) * candidate_state + torch.mul(update_gate,x2)
        x2_state = self.activate(x2_state+x2)
        return x2_state
 
class Gate_Update2(nn.Module):
    def __init__(self,attn_blocks,attn,in_channels, out_channels,d_model,
                heads,ranks,win,
                rank=False,dropout=0.1):
        super().__init__()
        '''out_channels==cross'''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn0 = attn_blocks(attn,in_channels+out_channels,in_channels+out_channels,d_model,heads,ranks,win,rank)
        self.attn1 = attn_blocks(attn,in_channels,out_channels,d_model,heads,ranks,win,rank)
        self.attn2 = attn_blocks(attn,out_channels,in_channels,d_model,heads,ranks,win,rank)
        self.attn3 = attn_blocks(attn,out_channels,out_channels,d_model,heads,ranks,win,rank)
        self.attn4 = attn_blocks(attn,in_channels+out_channels,in_channels+out_channels,d_model,heads,ranks,win,rank)
       
        self.conv1 = nn.Conv1d(in_channels+2*out_channels,out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(in_channels+out_channels,out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.activate = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tahn = nn.Tanh()
        # self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.norm = nn.LayerNorm(d_model)
    def forward(self, x1, x2):
        ''' x2==cross'''
        attn_cross_x2 = self.attn1(x1,x2) # the most relevant extra-interaction infor with x2
        attn_cross_x2_rev = self.attn2(x2,x1) # the most relevant extra-interaction infor with x2
        attn_self_x2 = self.attn3(x2) # the most relevant entra-interaction infor in x2
        attn_x2_new = torch.cat([attn_cross_x2,attn_cross_x2_rev,attn_self_x2],dim=1)
        reset_x2 = self.norm1(self.conv1(attn_x2_new))
        #
        comb1 = torch.cat([x1,x2],dim=1)
        attn_comb1 = self.attn0(comb1) # the most relevant extra-interaction infor with x2
        reset_x2 = attn_comb1[:, self.in_channels:,:] #x2
        ##
        comb2 = torch.cat([x1,reset_x2],dim=1)
        cand_attn_cross = self.attn4(comb2)
        cand_attn_cross_x1 = self.attn4(reset_x2,x1)
        cand_attn_cross = torch.cat([cand_attn_cross_x1,cand_attn_cross_x2],dim=1)
        out = self.norm2(self.conv2(cand_attn_cross)+x2)
        
        cand_attn_self = self.tahn(self.attn4(med_eff))
        balance the most relevant extra-interaction with x1 and the most relevant entra-interaction with med_eff
        out = (1-self.activate(cand_attn_cross))*x2 + self.activate(cand_attn_cross)*cand_attn_self  ##
        out = self.norm(self.conv(cand_attn_cross)+x2)
        out = self.activate(out)
        return out 

class Down_Mome(nn.Module):
    def __init__(self,Dps,GRUs,d_model):
        super().__init__()
        self.dps = nn.ModuleList(Dps)
        self.grus = nn.ModuleList(GRUs)
        self.h_dim = d_model//2
    def forward(self,x):
        bs = x.size(0)
        device = x.device
        h0 = torch.zeros([bs,self.h_dim]).to(device)
        pre = [h0]
        for i,(dp, gru) in enumerate(zip(self.dps, self.grus)):
            x = dp(x)
            for t in range(x.size(1)):
                xs = x[:,t]
                hd_ = gru(xs,pre[i])
            pre.append(hd_)
        return pre
 
class Up_Mome(nn.Module):
    def __init__(self,STs,RNNs,Projection,num_layers,d_model):
        super().__init__()
        self.num_layers = num_layers
        self.h_dim = d_model//2
       
        #self.ups = nn.ModuleList(Ups)
        self.st = STs
        self.rnns = nn.ModuleList(RNNs)
        self.pj = Projection
        #self.active = active
    def forward(self,x,stoken):
        bs = x.size(0)
        #hp_pre=[pre[-1]]
        #device = x.device
        #h0 = torch.zeros([1,bs,self.h_dim]).to(device)
        h0 = self.st(stoken.reshape(bs,-1))
        h0 = h0.unsqueeze(0)#.repeat(self.num_layers, 1, 1) 
        #print(h.shape)
        h_ls=[h0]
        #xs =[]
        for i,rnn in enumerate(self.rnns):
            x,hp_pre_ = rnn(x,h_ls[i])
            #x_ = pj(hp_pre_.view(bs,-1))
            h_ls.append(hp_pre_)
            #xs.append(x_)
            #x =  pj(hp_pre_).unsqueeze(-1)
     
            # x = up(hp_pre_.unsqueeze(-1))
            #x = pj(x.view(bs,-1)).unsqueeze(-1)
        #out = self.pj(h_ls[-1].reshape(bs,-1)).unsqueeze(-1)

            # print(x.shape)
            #xs.append(x)
        #out = self.active(x)
        #outs = torch.cat(h_ls[1:],dim=-1)
        outs = self.pj(h_ls[-1].reshape(bs,-1)).unsqueeze(-1)
        return outs
        


    

class Attn_Gate_Update1(nn.Module):
    def __init__(self,cross_attn,channels,d_model):
        super().__init__()
        '''out_channels==cross'''
        #self.attn1 = attn
        self.attn = cross_attn
        
        self.conv1 = nn.Conv1d(2*channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(2*channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model)
        # self.conv3 = nn.Conv1d(in_channels+out_channels, in_channels, kernel_size=3, padding=1)
        # self.norm3 = nn.LayerNorm(d_model)
        self.activate = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x1, x2):
        ''' x2==cross'''

        x2_attn = x2
        cross_attn,scores = self.attn(x2,x1,x1)
        # #
        med_comb = torch.cat([x2_attn,cross_attn],dim=1) ## [x1,x1]
        indir_attn_comb = self.norm1(self.conv1(med_comb))
        indir_gate = self.activate(indir_attn_comb)
        reset_comb = torch.mul(indir_attn_comb, indir_gate)
        med_comb_cand = torch.cat([reset_comb,x2_attn],dim=1) ##
        indir_attn_comb_new = self.conv2(med_comb_cand) #(x2,x1) 
        out = self.norm2(indir_attn_comb_new)
        update_gate = self.activate(out)
        out = (1-update_gate)*out + update_gate*cross_attn
        #
        out += cross_attn
        out = self.activate(out)
        return out

class Attn_Gate_Update2(nn.Module):
    def __init__(self,cross_attn_blocks,self_attn_blocks
                 #,channels,d_model,
                #heads,factor,ranks,win,attn_type,mask_flag,dropout
                 ):
        super().__init__()
        '''out_channels==cross'''
        #self.channels = channels
        self.attn1 =cross_attn_blocks
        self.attn2 =self_attn_blocks
        # self.activate = nn.Sigmoid()
        # self.relu = nn.ReLU()
        # self.conv1 = nn.Conv1d(2*channels, channels, kernel_size=3, padding=1)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        # self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x1, x2):
        ''' x2==cross'''
        # cross_attn, scores1 = self.attn1(x1,x2,x2)
        x1_attn, scores2 = self.attn2(x1,x1,x1)
        cross_attn, scores1 = self.attn1(x1_attn,x2,x2)
        # med_comb = torch.cat([x2_attn,cross_attn],dim=1) ## [x1,x1]
        # #
        # out = self.norm1(self.conv1(med_comb))
        # out += x2_attn
        # out = self.activate(out)
        return cross_attn
    
class FeedForwardAdapter(nn.Module):
    def __init__(self, channels1, channels2,d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(channels1, channels2, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        #self.bn1 = nn.BatchNorm1d(channels2)
        self.activate = nn.ReLU()
        self.conv2 = nn.Conv1d(channels2, channels1, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(d_model)
        #self.bn2 = nn.BatchNorm1d(channels1)
    def forward(self,x, cross=None):
        ##
        out = self.conv1(x)
        if cross is not None:
            out = self.norm1(out+cross)
        else:
            out = self.norm1(out)
        out = self.activate(out)
        ##
        out = self.conv2(out)
        out = self.norm2(out)
        out += x  
        #out = self.activate(out)
        return out

def enc_adas(encoder, adas):
    original_forward = encoder.forward
    encoder.adapters = nn.ModuleList(adas)
    def forward_with_adas(x,attn_mask=None):
        B, L, D =x.shape
        attns = []
        if encoder.conv_layers is not None:
            for i,(attn_layer, conv_layer) in enumerate(zip(encoder.attn_layers, encoder.conv_layers)):  
                x, attn = attn_layer(x,x,x, attn_mask=attn_mask)
                if i < len(encoder.adapters):
                    x = encoder.adapters[i](x)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = encoder.attn_layers[-1](x,x,x,attn_mask=attn_mask)
            x = encoder.adapters[-1](x)
            #x = x_lora+x
            attns.append(attn)
        else:
            for i,attn_layer in enumerate(encoder.attn_layers):
                x, attn = attn_layer(x,x,x, attn_mask=attn_mask)
                if i < len(encoder.adapters):
                    x = encoder.adapters[i](x)
                    #x = x_lora+x
                attns.append(attn)

        if encoder.norm is not None:
            x = encoder.norm(x)

        return x, attns
    # Replace the original forward method with the new one
    encoder.forward=forward_with_adas 

def dec_adas(decoder, adas,d_model,c_out):
    original_forward = decoder.forward
    decoder.adapters = nn.ModuleList(adas)
    decoder.projection_final = nn.Linear(d_model, c_out, bias=True)
    def forward_with_adas(x, cross):
        pre = [cross]
        for i, (module_ag, module_up, module_em, module_aa) in enumerate(zip(decoder.AG_Blocks, decoder.UP_Blocks, decoder.EM_Blocks, decoder.AA_Blocks)):
            pre_ = pre[-1]
            cross_ = module_ag(x, pre_)
            x = module_up(x)
            x = module_em(x)
            x = module_aa(x, cross_)
            # Apply corresponding adapter after each AA_Block
            if i < len(decoder.adapters):
                x= decoder.adapters[i](x)
            pre.append(cross_)

        if decoder.projection is not None:
            x = decoder.projection_final(x)
        return x
    # Replace the original forward method with the new one
    decoder.forward=forward_with_adas 

class UNetBlock(nn.Module):
    def __init__(self, in_channels, stride_g, scale_factor,d_model, down=False, gene=False, dropout=0.2):
        super(UNetBlock, self).__init__()
        self.scale_factor = scale_factor
        self.down = down
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.LayerNorm(d_model)
        if down:
            self.block = ConvMaxpool(in_channels, stride_g,d_model)  
        else:
            self.block = InterpolateUpsample(in_channels, scale_factor,d_model)
    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        return out


