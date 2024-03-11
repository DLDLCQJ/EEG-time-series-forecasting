


#%%
net = Creating_h0()
net.to(device)
#print(net)
#train_xht = torch.from_numpy(train_xh).float()
h0_tr = net(train_xt.to(device))

loss_BCE = torch.nn.BCELoss(reduction = 'mean')
loss_MSE = torch.nn.MSELoss(reduction = 'mean')
#log_var = torch.log(sigma**2)
loss_KLD = lambda mu,log_var: (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))/torch.numel(mu.data)


def pretrain_epoch(optim_dict,model_dict, y,loss_BCE,latent_dim):
    loss_dict = {'G': [], 'D': []}
    batch_size=y.shape[0]
    ###GAN###########

    ###D### 
    optim_dict["D"].zero_grad()
    random_noise = torch.randn(batch_size, latent_dim).to(device)
    fake_data = model_dict["G"](random_noise.reshape(batch_size,latent_dim,1)) #z:[60, 64,1];fake_data:[60, 2267]
    #print(fake_data.shape)

    fake_data = torch.cat([y[:, :6801, :], fake_data.reshape(-1, 6801//3, 1)], axis = 1).to(device)#torch.Size([60, 9068,1])
    ###
    critic_real = model_dict["D"](y) #
    #real_labels = torch.ones_like(critic_real).to(device)
    #dis_real_loss = loss_BCE(critic_real, real_labels)
    
    critic_fake = model_dict["D"](fake_data.detach())
    #fake_labels =  torch.zeros_like(real_labels).to(device)
    #dis_fake_loss = loss_BCE(critic_fake, fake_labels)
    #lossD = dis_real_loss + dis_fake_loss
    gp = calc_gradient_penalty(model_dict["D"], y.data, fake_data.data)
    lossD = -torch.mean(critic_real) + torch.mean(critic_fake) + gp
    lossD.backward() #retain_graph = True
    optim_dict["D"].step()
    #
    loss_dict["D"].append(lossD.data.item())
    
    ###G####
    optim_dict["G"].zero_grad()
    output_fake = model_dict["D"](fake_data)
    #lossG = loss_BCE(output_fake, real_labels)
    lossG = -torch.mean(output_fake) ##

    lossG.backward()
    optim_dict["G"].step()
    loss_dict["G"].append(lossG.data.item())
    
    
    return fake_data, loss_dict


