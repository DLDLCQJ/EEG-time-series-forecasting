

def init_model_dict(input_dim,h_dim,d_model,num_layers,tw,pred, head, rank):
    model_dict = {}
    model_dict["E"] = Encoder(input_dim, h_dim,d_model,num_layers,tw, head,rank)
    #model_dict["DL"] = Decoder_LATN(input_dim,latent_dim,h_dim,num_layers, tw, head,rank)
    model_dict["DA"] = Decoder_ATTN(input_dim,h_dim,d_model,num_layers, tw, head,rank)
    model_dict["G"] = AttnGenerator(input_dim,h_dim, d_model,pred, head,rank)

    model_dict["D"] = AttnDiscriminator(input_dim, pred+h_dim,head,rank) 
    model_dict["F"] = Attnfinetuning(input_dim, tw,pred,d_model) 
    return model_dict

def init_optim_dict(lr):
    optim_dict = {}
    #optim_dict["DL"] = optim.Adam(list(model_dict["E"].parameters())+list(model_dict["DL"].parameters()), lr=lr)
    optim_dict["DA"] = optim.Adam(list(model_dict["E"].parameters())+list(model_dict["DA"].parameters()), lr=lr)
    optim_dict["G"] = optim.Adam(model_dict["G"].parameters(), lr=lr)
    optim_dict["D"] = optim.Adam(model_dict["D"].parameters(), lr=lr)
    optim_dict["F"] = optim.Adam(model_dict["F"].parameters(), lr=lr) 
    return optim_dict
