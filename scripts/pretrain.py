def pretrain_epoch(optim_dict,model_dict, y,d_model,pred,tw):
    batch_size=y.shape[0]
    ###D### 
    optim_dict["D"].zero_grad()
    random_noise = torch.randn(batch_size, d_model).to(device)
    g_out = model_dict["G"](random_noise.reshape(1,batch_size,d_model)) #z:[60, 64,1];fake_data:[60, 2267]
    #fake_data_cat = torch.cat([y[:,-tw:,:], fake_data[0].reshape(-1, pred, 1)], axis = 1).to(device)#torch.Size([60, 9068,1])
    critic_real = model_dict["D"](y) 
    critic_fake = model_dict["D"](g_out.detach())
    gp = calc_gradient_penalty(model_dict["D"], y.data, g_out[0].data)
    loss_D = -torch.mean(critic_real) + torch.mean(critic_fake) + gp
    loss_D.backward() 
    optim_dict["D"].step()
    ###G####
    optim_dict["G"].zero_grad()
    output_fake = model_dict["D"](g_out)
    loss_G = -torch.mean(output_fake) ##
    loss_G.backward()
    optim_dict["G"].step()
    
    return  loss_D, loss_G,fake_data
