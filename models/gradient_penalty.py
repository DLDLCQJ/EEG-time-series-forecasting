
def calc_gradient_penalty(Disc, real_data, fake_data, LAMBDA=0.2):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    t = torch.rand(real_data.size(0),1,1)
    epsilon = t.expand_as(real_data).to(device)
    #print(alpha.shape)
    # Get random interpolation between real and fake samples: (60,2267)
    interpolates = (epsilon * real_data + ((1 - epsilon) * fake_data)).requires_grad_(True)
    ##
    disc_interpolates = Disc(interpolates)
    fake = torch.ones_like(disc_interpolates).to(device)
    # Get gradient w.r.t. interpolates
    grads = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    #grads = grads.view(grads.size(0), -1)
    #grad_penalty = torch.pow((grads.norm(2, dim=1) - 1), 2).mean()
    #The torch.norm can cause similar problems (becase of the square root), so its better to calculate it manually with epsilon:
    #grad_penalty = torch.mean((1 - torch.sqrt(1e-8+torch.sum(grads**2, dim=1)))**2)
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return grad_penalty
