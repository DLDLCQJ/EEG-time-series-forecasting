loss_BCE = torch.nn.BCELoss(reduction = 'mean')
loss_MSE = torch.nn.MSELoss(reduction = 'mean')
loss_KLD = lambda mu,log_var: (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))/torch.numel(mu.data)
