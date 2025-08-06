from utils import add_noise
import torch
import numpy as np    
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from losses import update_lam, loss_L2



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in monitored metric to be considered improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if validation loss improves
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False  # Continue training

 
def train(model, train_loader, loss_func, optimizer, device, epochs=50, do_add_noise=True, **kwargs):

    min_noise = kwargs.get('min_noise', 0.4)
    max_noise = kwargs.get('max_noise', 0.4)

    model.train()
    train_losses = []
    early_stopping = EarlyStopping(patience=8, min_delta=0.00)
    for epoch in range(epochs):
       
        epoch_loss = 0.0

        for data, _ in train_loader:
            # Add noise to input images
            noise_level = np.random.uniform(min_noise, max_noise)
            if do_add_noise:
                noisy_data = add_noise(data,noise_factor=noise_level).to(device)
            else:
                noisy_data = data.to(device)
            clean_data = data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass

           
            # Compute loss
            loss = loss_func(noisy_data, clean_data, model, epoch, **kwargs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break  # Stop training
        train_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    return train_losses

def train_al(model, train_loader, loss_func, optimizer, device, epochs=50, do_add_noise=True, **kwargs):

    do_update = kwargs.get('do_update', True)
    lam_vec = kwargs.get('lam3', 1)
    min_noise = kwargs.get('min_noise', 0.4)
    max_noise = kwargs.get('max_noise', 0.4)
    train_losses = []
    lam_list=[]
    early_stopping = EarlyStopping(patience=8, min_delta=0.00)
   
    model.train()


    for epoch in range(epochs):
       
        epoch_loss = 0.0

        for data, _ in train_loader:
            noise_level = np.random.uniform(min_noise, max_noise)
            # Add noise to input images
            if do_add_noise:
                noisy_data = add_noise(data,noise_factor=noise_level).to(device)
            else:
                noisy_data = data.to(device)
            clean_data = data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass

           
            # Compute loss
            loss,loss_vector = loss_func(noisy_data, clean_data, model, epoch, lam_vec = lam_vec, **kwargs)
           
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            #lam_vec = update_lam(loss_vector)
            epoch_loss += loss.item()
           

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break  # Stop training
        train_losses.append(avg_loss)
        if do_update:
            lam_vec = update_lam(loss_vector)
        lam_list.append(lam_vec)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f} lam3={lam_vec:.4f} ')
        
      
    return train_losses,lam_list,loss_vector


def train_diffusion(dae_model, diffusion_model, dataloader, optimizer, epoch, device, diffusion_steps = 1200, digit=None):


    betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
  

    diffusion_model.train()
    dae_model.eval()
    for param in dae_model.parameters():
        param.requires_grad = False
    train_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        # Get latent representations from DAE
        with torch.no_grad():
            latents = dae_model.encoder(data)
            latents = latents[0] if isinstance(latents, tuple) else latents
        # Sample timesteps
        t = torch.randint(0, diffusion_steps, (data.shape[0],), device=device)

        # Add noise to the latents according to the timestep
        noisy_latents, noise = q_sample(latents, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        # Predict the noise
        optimizer.zero_grad()
        noise_pred = diffusion_model(noisy_latents, t)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # if batch_idx % 50 == 0:
        #     digit_str = f" Digit {digit}" if digit is not None else ""
        #     print(f'Diffusion Train{digit_str} Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
        #            f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(dataloader)
    
    digit_str = f" Digit {digit}" if digit is not None else ""
    print(f'====> Diffusion{digit_str} Epoch: {epoch} Average loss: {avg_loss:.6f}')
    return avg_loss

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def train_dae_diffusion_model(digit, digit_train_loader, dae_model, diffusion_model, device, dae_epochs, diff_epochs,lr=1e-3,
                          diffusion_steps=1200, dae_model_path=None,diff_model_path=None, min_noise=0.4, max_noise=0.4):
    print(f"Training models for digit {digit}...")

   
    # Optimizers
    dae_optimizer = optim.Adam(dae_model.parameters(), lr=lr)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

     # Train DAE
    print(f"Training DAE for digit {digit}...")
    dae_losses = train(dae_model, digit_train_loader, loss_L2, dae_optimizer, device, epochs=dae_epochs,
                       do_add_noise=True,min_noise=min_noise, max_noise=max_noise)
    torch.save(dae_model.state_dict(), dae_model_path)
    
   
    # Train diffusion model
    print(f"Training Diffusion for digit {digit}...")
    diffusion_losses = []
    min_loss = float('inf')
    early_stopping = EarlyStopping(patience=5, min_delta=0.00)

    for epoch in range(1, diff_epochs + 1):
        loss = train_diffusion(dae_model, diffusion_model, digit_train_loader, 
                               diffusion_optimizer, epoch, device, diffusion_steps = diffusion_steps,digit=digit)
       

        if early_stopping(loss):
            print(f"Early stopping at epoch {epoch+1}")
            break  # Stop training
        diffusion_losses.append(loss)
        # if loss < min_loss:
        #     min_loss = loss
        #     torch.save(diffusion_model.state_dict(),diff_model_path)

    print('saving {}'.format(diff_model_path))       
    torch.save(diffusion_model.state_dict(),diff_model_path)
    

    # Plot losses
    # plt.figure(figsize=(10, 5))
    # plt.plot(dae_losses, label=f'DAE Loss (Digit {digit})')
    # plt.plot(diffusion_losses, label=f'Diffusion Loss (Digit {digit})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title(f'Training Losses for Digit {digit}')
    # plt.savefig(f'losses_digit_{digit}.png')
    # plt.close()

   
    # Generate final samples
    

    # checkpoint = torch.load(f'Models/dae_digit{digit}.pth',weights_only=True)
    # dae.load_state_dict(checkpoint)
    # checkpoint = torch.load(f'Models/diffusion_digit{digit}.pth',weights_only=True)
    # diffusion.load_state_dict(checkpoint)

    # dae.eval()
    # diffusion.eval()

    # print(f"Generating final samples for digit {digit}...")
    # samples, _ = p_sample_loop(diffusion, dae, shape=(batch_size, latent_dim), n_samples=16)

    # os.makedirs('Samples', exist_ok=True)
    # # Plot a grid of samples
    # fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    # axs = axs.flatten()

    # for i, ax in enumerate(axs):
    #     if i < len(samples):
    #         img = samples[i].cpu().squeeze().numpy()
    #         ax.imshow(img, cmap='gray')
    #         ax.axis('off')

    # plt.tight_layout()
    # plt.savefig(f"Samples/final_samples_digit_{digit}.png")
    # plt.close()

    return dae_model, diffusion_model

def train_all(cfg,AE_model,dnet_model,train_loader,model_path,figure_path,device,**kwargs):

    
    
    AE_model.train()
    dnet_model.train()
    all_images=[]
    for images, _ in train_loader:
        all_images.append(images)

    # Concatenate all batches along the first dimension (batch dimension)
    all_images = torch.cat(all_images, dim=0)
    all_images = all_images.to(device)

    optimizer = optim.Adam([
        {'params': AE_model.parameters(), 'lr': cfg['lr_ae']},
        {'params': dnet_model.parameters(), 'lr': cfg['lr_dnet']}
        ])
   
    train_losses,lam_list, loss_vector = train_al(AE_model, train_loader, cfg['loss_func'], optimizer, 
                        device, epochs=cfg['epochs'], dnet= dnet_model,all_images=all_images,
                        lam3=cfg['lam3'],lam6=cfg['lam6'],lam7=cfg['lam7'],do_update=cfg['do_update'],**kwargs) 
    checkpoint = {
        'AE_state_dict': AE_model.state_dict(),
        'dnet_state_dict': dnet_model.state_dict()}
    
    
    
    print('saving {}'.format(model_path))
    torch.save(checkpoint, model_path )
    plt.figure()
    plt.plot(train_losses)
    plt.title('train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(figure_path,f'train_loss.png'))
    plt.close()  