
def train_epoch(optim_dict,model_dict,x,h0,y,loss_MSE,loss_KLD,latent_dim):    
    #####VAE########
    loss_dict = {'V': [], 'EG': []}
    optim_dict["V"].zero_grad()
    vae_out = model_dict["V"](x,h0) #!!!
    loss_re = loss_MSE(vae_out[0], x) 
    loss_norm = loss_KLD(vae_out[2], vae_out[3]) 
    loss_V = loss_re + loss_norm
    loss_V.backward()
    optim_dict["V"].step()
    loss_dict["V"].append(loss_V.data.item())

    #EG
    optim_dict["EG"].zero_grad()
    pred_data = model_dict["G"](model_dict["V"].encoder(x, h0)[1].reshape(batch_size,latent_dim,1)) ##mean
    pred_data = torch.cat([y[:, :6801, :], pred_data.reshape(-1, 6801//3, 1)], axis = 1)
    loss_EG = loss_MSE(pred_data, y)
    loss_EG.backward()
    optim_dict["EG"].step()
    loss_dict["EG"].append(loss_EG.data.item())
    return loss_dict, pred_data
