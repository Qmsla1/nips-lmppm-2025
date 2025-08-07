import torch
from utils import grad_d_z

def loss_L2(noisy_data, clean_data, model, epoch, **kwargs):
    reconstructed = model(noisy_data)
    loss =  torch.mean(torch.square(reconstructed-clean_data))
    return loss

def loss_L1(noisy_data, clean_data, model, epoch, **kwargs):
    reconstructed = model(noisy_data)
    loss =  torch.mean(torch.abs(reconstructed-clean_data))
    return loss
def loss_ssim(noisy_data, clean_data, model, epoch, **kwargs):
    ssim_loss = kwargs['ssim_loss']
    reconstructed = model(noisy_data)
    loss =  torch.mean(torch.abs(ssim_loss(reconstructed,clean_data)))
    return loss

def loss_dist(noisy_data, clean_data, model, epoch, **kwargs):

    k = 1  # Set k to your desired value
    warm_up_epochs=-1

    lam1= kwargs.get('lam1', 1)
    lam2= kwargs.get('lam2', 1)
    lam3= kwargs.get('lam3', 0.3)
    lam4= kwargs.get('lam4', 1)
    lam5= kwargs.get('lam5', 1)
    lam6= kwargs.get('lam6', 0.3)
    lam7= kwargs.get('lam7', 0.3)
    eps = kwargs.get('eps', 1e-10) 
    #all_images = kwargs['all_images']
    #all_z_clean = model.encoder(all_images)



    dnet = kwargs['dnet']
    z_clean = model.encoder(clean_data).requires_grad_(True)
    z = model.encoder(noisy_data).requires_grad_(True)
    
    x_in_M = clean_data
    x_notin_M = noisy_data
    
    x_hat_in_M = model(x_in_M)
    x_hat_notin_M = model(x_notin_M)

    d_in_M = dnet(z_clean)
    d_notin_M = dnet(z)

    #grad_d_in_M , grad_d_norm_in_M = grad_d_z(z_clean, d_in_M)
    grad_d_notin_M , grad_d_norm_notin_M = grad_d_z(z, d_notin_M,eps=eps)

    if epoch > 1e+7:#:warm_up_epochs:
        pairwise_dists = torch.cdist(z, all_z_clean, p=2)
        #topk_values, _ = torch.topk(pairwise_dists, k=k, dim=1, largest=False)
        #dist_notin_M = torch.mean(topk_values, dim=1)
        dist_notin_M = torch.min(pairwise_dists, dim=1)[0]
        #dist_notin_M,idx = torch.min(pairwise_dists, dim=1)
        
    else:
        dist_notin_M = torch.norm(z - z_clean, p=2, dim=(1))



   
    loss1 = torch.mean(torch.square(d_notin_M - dist_notin_M.reshape(-1, 1)))
    loss2 = torch.mean(torch.square(x_hat_in_M - clean_data))
    loss3 = torch.mean(torch.square(grad_d_norm_notin_M - 1))
    loss4 = torch.mean(torch.square(d_in_M))
    loss5 = torch.mean(torch.square(torch.abs(d_notin_M) - d_notin_M)) \
            + torch.mean(torch.square(torch.abs(d_in_M) - d_in_M))

    
    d = d_notin_M
    z_shift = z - d * grad_d_notin_M
    
    if epoch > 1e+7:#warm_up_epochs:
        pairwise_dists = torch.cdist(z_shift, all_z_clean, p=2)
        dists = torch.min(pairwise_dists, dim=1)[0]
        #min_values, min_indices = torch.min(pairwise_dists, dim=1)
        loss6 = torch.mean(dists)
    else:
        loss6 = torch.mean(torch.abs((z_clean) - (z_shift))) 

    G_z_shift = model.decoder(z_shift)
    if epoch > 1e+7:#warm_up_epochs:
        G_z_shift_flat = G_z_shift.view(G_z_shift.size(0), -1)
        all_images_flat = all_images.view(all_images.size(0), -1)
        pairwise_dists = torch.cdist(G_z_shift_flat,  all_images_flat, p=2)
        dists = torch.min(pairwise_dists, dim=1)[0]
        loss7 = torch.mean(dists)
    else:
        loss7 = torch.mean(torch.abs((clean_data) - G_z_shift)) 

    loss = lam1*loss1 + lam2*loss2 + lam3*loss3 + lam4*loss4 +lam5*loss5 + lam6*loss6 + lam7*loss7
   
    # print('loss1:',loss1)
    # print('loss2:',loss2)
    # print('loss3:',loss3)
    # print('loss4:',loss4)
    # print('loss5:',loss5)
    # print('loss6:',loss6)
    # print('loss7:',loss7)
    return loss



def update_lam(loss_vec):
    #Adaptive Reweighting

    

    lam3 = max(loss_vec[0]/loss_vec[2],1e-3)
   
    

    return lam3


def loss_dist_al(noisy_data, clean_data, model, epoch, lam_vec, **kwargs):

   

    lam1= kwargs.get('lam1', 1)
    lam2= kwargs.get('lam2', 1)
    lam3= kwargs.get('lam3', 1)
    lam4= kwargs.get('lam4', 1)
    lam5= kwargs.get('lam5', 1)
    lam6= kwargs.get('lam6', 1.0)
    lam7= kwargs.get('lam7', 1.0)
    lam8= kwargs.get('lam8', 1.0)
    n_points = kwargs.get('n', 5)
    eps = kwargs.get('eps', 1e-10)
   

    lam3 = lam_vec
    



    dnet = kwargs['dnet']

    ####
    clean_images = kwargs['all_images']  
    noisy_flat = noisy_data.view(noisy_data.shape[0], -1)
    clean_flat = clean_images.view(clean_images.shape[0], -1)  
    pairwise_dists = torch.cdist(noisy_flat, clean_flat, p=2)
    _, min_indices = torch.min(pairwise_dists, dim=1)
    x_star = clean_images[min_indices]

    #z_star = model.encoder(x_star).requires_grad_(True)
    latents = model.encoder(x_star)
    z_star = latents[0] if isinstance(latents, tuple) else latents
    #z_star = z_star.requires_grad_(True)
    ###

    ####Test
    # x_star = clean_data
    # z_star = model.encoder(x_star).requires_grad_(True)
    ####

    latents = model.encoder(clean_data)
    z_clean = latents[0] if isinstance(latents, tuple) else latents
    #z_clean = model.encoder(clean_data).requires_grad_(True)
    latents = model.encoder(noisy_data)
    z = latents[0] if isinstance(latents, tuple) else latents
    #latents_star = model.encoder(x_star)
    skip_features = latents[1] if isinstance(latents, tuple) else None
    #z = model.encoder(noisy_data).requires_grad_(True)
    z.requires_grad_(True)

    

    x_in_M = clean_data
    x_notin_M = noisy_data
    
    x_hat_in_M = model(x_in_M)
   

    d_in_M = dnet(z_clean)
    d_notin_M = dnet(z)

   
    grad_d_notin_M , grad_d_norm_notin_M = grad_d_z(z, d_notin_M,eps=eps)
    #dist_notin_M = torch.norm(z - z_clean, p=2, dim=(1))
    dist_notin_M = torch.norm(z - z_star, p=2, dim=(1))



   
    loss1 = torch.mean(torch.square(d_notin_M - dist_notin_M.reshape(-1, 1)))
    loss2 = torch.mean(torch.square(x_hat_in_M - clean_data))
    loss3 = torch.mean(torch.square(grad_d_norm_notin_M - 1))
    loss4 = torch.mean(torch.square(d_in_M))
    loss5 = torch.mean(torch.square(torch.abs(d_notin_M) - d_notin_M)) \
            + torch.mean(torch.square(torch.abs(d_in_M) - d_in_M))

    
    d = d_notin_M
    z_shift = z - d * grad_d_notin_M
    #loss6 = torch.mean(torch.abs((z_clean) - (z_shift))) 
    loss6 = torch.mean(torch.abs((z_star) - (z_shift))) 

    if skip_features is not None:
        G_z_shift = model.decoder(z_shift, skip_features)
    else:
        G_z_shift = model.decoder(z_shift)
    #loss7 = torch.mean(torch.abs((clean_data) - G_z_shift)) 
    loss7 = torch.mean(torch.abs((x_star) - G_z_shift))

    # Sample additional points outside the manifold and compute pairwise distances
    loss8 = torch.tensor(0.0, device=noisy_data.device)
    if n_points > 1:
        z_interp = []
        for idx in range(1, n_points + 1):
            alpha = idx / 10.0
            x_in = (1 - alpha) * x_notin_M + alpha * x_star
            lat = model.encoder(x_in)
            z_in = lat[0] if isinstance(lat, tuple) else lat
            z_interp.append(z_in)
        z_interp = torch.stack(z_interp, dim=1)  # [batch, n_points, latent_dim]
        diffs = z_interp.unsqueeze(1) - z_interp.unsqueeze(2)
        sq_dists = torch.sum(diffs**2, dim=-1)
        iu = torch.triu_indices(n_points, n_points, offset=1)
        loss8 = torch.mean(sq_dists[:, iu[0], iu[1]])

    loss = (lam1*loss1 + lam2*loss2 + lam3*loss3 + lam4*loss4 + lam5*loss5 +
            lam6*loss6 + lam7*loss7 + lam8*loss8)

    loss_vector = [loss1.item(), loss2.item(), loss3.item(), loss4.item(),
                   loss5.item(), loss6.item(), loss7.item(), loss8.item()]

    return loss,loss_vector

def loss_dist_old(noisy_data, clean_data, model, **kwargs):

    lam1=1
    lam2=1
    lam3=1.0
    lam4=0.5#0.5
    lam5=1
    lam6=1
    lam7=0.5
    lam8=0.5

    eps = kwargs.get('eps', 1e-10) 

   

    dnet = kwargs['dnet']

    z_clean = model.encoder(clean_data).requires_grad_(True)
    z = model.encoder(noisy_data).requires_grad_(True)
    
    x_in_M = clean_data
    x_notin_M = noisy_data
    
    x_hat_in_M = model(x_in_M)
    x_hat_notin_M = model(x_notin_M)

    d_in_M = dnet(z_clean)
    d_notin_M = dnet(z)

    #grad_d_in_M , grad_d_norm_in_M = grad_d_z(z_clean, d_in_M)
    grad_d_notin_M , grad_d_norm_notin_M = grad_d_z(z, d_notin_M,eps=eps)

   
    dist_notin_M = torch.norm(z - z_clean, p=2, dim=(1))

    

    #loss1 = torch.mean(torch.square(d_in_M - dist_in_M.reshape(-1, 1)))
    loss1 = 0 
    loss2 = torch.mean(torch.square(d_notin_M - dist_notin_M.reshape(-1, 1)))
    #loss3 = torch.mean(torch.square(x_hat_notin_M - clean_data))
    loss3 = torch.mean(torch.square(x_hat_in_M - clean_data))
    loss4 = torch.mean(torch.square(grad_d_norm_notin_M - 1))
    loss5 = torch.mean(torch.square(d_in_M))
    loss6 = torch.mean(torch.square(torch.abs(d_notin_M) - d_notin_M)) \
            + torch.mean(torch.square(torch.abs(d_in_M) - d_in_M))

    
    d = d_notin_M
    z_shift = z - d * grad_d_notin_M
    

    loss7 = torch.mean(torch.abs((z_clean) - (z_shift))) 
    loss8 = torch.mean(torch.abs((clean_data) - model.decoder(z_shift))) 

    loss = lam1*loss1 + lam2*loss2 + lam3*loss3 + lam4*loss4 +lam5*loss5 + lam6*loss6 + lam7*loss7+lam8*loss8
    # print('loss1:',loss1)
    # print('loss2:',loss2)
    # print('loss3:',loss3)
    # print('loss4:',loss4)
    # print('loss5:',loss5)
    # print('loss6:',loss6)
    # print('loss7:',loss7)
    return loss

def loss_prob(noisy_data, clean_data, model, **kwargs):


    z = model.encoder(noisy_data)
    reconstructed = model.decoder(z)
    z_mean = kwargs['z_mean']
    inv_cov = kwargs['inv_cov']
   
    lam = kwargs['lam']
    z_centered = z - z_mean
    loss_einsum = torch.einsum('bp,pq,bq->b', z_centered, inv_cov, z_centered)
    mahal_loss = torch.mean(loss_einsum)

    loss = torch.mean(torch.square(reconstructed-clean_data))+lam*mahal_loss
    return loss