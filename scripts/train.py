
def train_epoch(optim_dict,model_dict,x,h,y,loss_MSE,loss_KLD,tw, pred):    
    bs=x.shape[0]
    optim_dict["DA"].zero_grad()
    optim_dict["F"].zero_grad()
    out_E = model_dict["E"](x,h)  
    recon_out_attn = model_dict["DA"](out_E)  
    g_out = model_dict["G"](out_E)
    pred_data = model_dict["F"](g_out) ##mean
    loss_re_attn = loss_MSE(recon_out_attn, x) 
    #loss_norm_attn = loss_KLD(out_E[0], out_E[1]) 
    loss_eg = loss_MSE(pred_data, y)
    DA_loss_attn = loss_re_attn
    DA_loss_attn.backward(retain_graph=True)
    loss_eg.backward()
    optim_dict["DA"].step()
    optim_dict["F"].step()

    return loss_eg, recon_out_attn,pred_data
