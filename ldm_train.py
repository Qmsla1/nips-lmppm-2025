import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import random
import os
import torchvision
from architectures import MNISTConvAutoencoder, DiffusionModel
from train import train,EarlyStopping
from losses import loss_L2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_steps = 1200
latent_dim = 13
batch_size = 128
betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
# Set random seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)
transform = transforms.Compose([
    transforms.ToTensor(),
])

def visualize_samples(samples, epoch, digit=None, path="samples"):
    import os
    digit_str = f"_digit_{digit}" if digit is not None else ""
    path = f"{path}{digit_str}"

    if not os.path.exists(path):
        os.makedirs(path)

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(samples):
            img = samples[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{path}/epoch_{epoch}.png")
    plt.close()




@torch.no_grad()
def p_sample(model, x, t):
    betas_t = betas[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    sqrt_recip_alphas_t = torch.rsqrt(alphas[t]).view(-1, 1)

    # Equation 11 in the DDPM paper
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t[0] == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t].view(-1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Complete reverse diffusion sampling from latent
@torch.no_grad()
def p_sample_loop(diffusion_model, dae_model, latent_start=None, shape=None, n_samples=8, diffusion_steps_used=None):
    device = next(diffusion_model.parameters()).device

    if diffusion_steps_used is None:
        diffusion_steps_used = diffusion_steps

    if latent_start is None:
        # Start from pure noise
        assert shape is not None, "Must provide shape if no latent_start"
        img = torch.randn(n_samples, latent_dim, device=device)
    else:
        # Start from provided latent
        img = latent_start
        n_samples = img.shape[0]

    imgs = []

    for i in reversed(range(0, diffusion_steps_used)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        img = p_sample(diffusion_model, img, t)
        imgs.append(img)

    # Decode the latent samples
    latent_samples = img
    decoded_samples = dae_model.decoder(latent_samples)

    return decoded_samples, imgs

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def train_diffusion(dae_model, diffusion_model, dataloader, optimizer, epoch, digit=None):
    diffusion_model.train()
    dae_model.eval()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        # Get latent representations from DAE
        with torch.no_grad():
            #_, latents = dae_model(data, add_noise=False)
            latents = dae_model.encoder(data)
        # Sample timesteps
        t = torch.randint(0, diffusion_steps, (data.shape[0],), device=device)

        # Add noise to the latents according to the timestep
        noisy_latents, noise = q_sample(latents, t)

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

def train_digit_models(digit, digit_train_loader, latent_dim, dae_epochs, diff_epochs,lr=1e-3):
    print(f"Training models for digit {digit}...")

    os.makedirs('Models', exist_ok=True)
    # Create digit-specific dataset
   
    # Initialize models
    dae = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
    diffusion = DiffusionModel(latent_dim).to(device)

    # Optimizers
    dae_optimizer = optim.Adam(dae.parameters(), lr=lr)
    diffusion_optimizer = optim.Adam(diffusion.parameters(), lr=lr)

     # Train DAE
    print(f"Training DAE for digit {digit}...")
    dae_losses = train(dae, digit_train_loader, loss_L2, dae_optimizer, device, epochs=dae_epochs,do_add_noise=True)
    torch.save(dae.state_dict(), f'Models/dae_digit{digit}.pth')
    
   
    # Train diffusion model
    print(f"Training Diffusion for digit {digit}...")
    diffusion_losses = []
    min_loss = float('inf')
    

    for epoch in range(1, diff_epochs + 1):
        loss = train_diffusion(dae, diffusion, digit_train_loader, diffusion_optimizer, epoch, digit=digit)
       
        diffusion_losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            torch.save(diffusion.state_dict(), f'Models/diffusion_digit{digit}.pth')
           
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(dae_losses, label=f'DAE Loss (Digit {digit})')
    plt.plot(diffusion_losses, label=f'Diffusion Loss (Digit {digit})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Losses for Digit {digit}')
    plt.savefig(f'losses_digit_{digit}.png')
    plt.close()

   
    # Generate final samples
    

    checkpoint = torch.load(f'Models/dae_digit{digit}.pth',weights_only=True)
    dae.load_state_dict(checkpoint)
    checkpoint = torch.load(f'Models/diffusion_digit{digit}.pth',weights_only=True)
    diffusion.load_state_dict(checkpoint)

    dae.eval()
    diffusion.eval()

    print(f"Generating final samples for digit {digit}...")
    samples, _ = p_sample_loop(diffusion, dae, shape=(batch_size, latent_dim), n_samples=16)

    os.makedirs('Samples', exist_ok=True)
    # Plot a grid of samples
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(samples):
            img = samples[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"Samples/final_samples_digit_{digit}.png")
    plt.close()

    return dae, diffusion


def main():
# Hyperparameters
   
   
    dae_epochs = 100#80
    diff_epochs = 100#50
    lr = 1e-3
   
   


    digits_to_train = range(10) 
    dae_list=list(range(10))
    diffusion_list=list(range(10))
    test_ds = list(range(10))
    test_loader_list = list(range(10))
    train_ds = list(range(10))
    train_loader_list = list(range(10))

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

    for digit in digits_to_train:

        idx = (test_dataset.targets == digit)
        test_ds[digit] = torch.utils.data.Subset(test_dataset, indices=torch.where(idx)[0])
        test_loader_list[digit] = torch.utils.data.DataLoader(test_ds[digit], batch_size=batch_size, shuffle=False)


        idx = (train_dataset.targets == digit)
        train_ds[digit] = torch.utils.data.Subset(train_dataset, indices=torch.where(idx)[0])
        train_loader_list[digit] = torch.utils.data.DataLoader(train_ds[digit], batch_size=batch_size, shuffle=True)

        
        dae, diffusion = train_digit_models(digit, train_loader_list[digit], latent_dim=latent_dim,
                                            dae_epochs=dae_epochs,diff_epochs=diff_epochs,lr=lr)
        dae_list[digit] = dae
        diffusion_list[digit] = diffusion
       

if __name__ == '__main__':
    main()