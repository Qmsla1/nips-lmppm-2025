
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

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
latent_dim = 15 #32
epochs = 50
lr = 1e-3
noise_factor = 0.3  # For denoising autoencoder
diffusion_steps = 1000
betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Random degradation transforms
class RandomDegradation(object):
    def __init__(self, degradation_type='rotation', severity=1):
        self.degradation_type = degradation_type
        self.severity = severity

    def __call__(self, img):
        if self.degradation_type == 'rotation':
            # Rotate the image by a random angle between -45 and 45 degrees
            angle = random.uniform(80 * self.severity, 110 * self.severity)
            return TF.rotate(img, angle)
        elif self.degradation_type == 'blur':
            # Apply Gaussian blur
            blur_kernel = int(4 * self.severity) * 2 + 1  # Ensure odd kernel size
            return TF.gaussian_blur(img, blur_kernel, sigma=1.0 * self.severity)
        elif self.degradation_type == 'noise':
            # Add random noise
            img_array = np.array(img)
            noise = np.random.normal(0, 25.5 * self.severity, img_array.shape).astype(np.uint8)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        elif self.degradation_type == 'crop':
            # Random crop and resize back
            width, height = img.size
            crop_size = int(width * (1 - 0.2 * self.severity))
            return TF.center_crop(img, crop_size)
        else:
            return img


# MNIST Dataset with degradation
class DegradedMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True,
                 degradation_type='rotation', severity=1):
        super(DegradedMNIST, self).__init__(root, train, transform, target_transform, download)
        self.degradation = RandomDegradation(degradation_type, severity)

    def __getitem__(self, index):
        img, target = super(DegradedMNIST, self).__getitem__(index)

        # Convert tensor to PIL for degradation
        if isinstance(img, torch.Tensor):
            # If tensor is [C,H,W], convert to PIL
            if img.dim() == 3:
                pil_img = TF.to_pil_image(img)
                degraded_img = self.degradation(pil_img)
                degraded_tensor = TF.to_tensor(degraded_img)
                return (degraded_tensor, img), target  # Return both degraded and original

        return img, target


# Function to create a digit-specific dataset
def create_digit_dataset(dataset, digit):
    """Filter the dataset to include only samples of the specified digit."""
    indices = []
    for i, (_, target) in enumerate(dataset):
        if target == digit:
            indices.append(i)
    return Subset(dataset, indices)


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create datasets
degradation_type = 'blur'  # Options: 'rotation', 'blur', 'noise', 'crop'
severity = 1.0  # Between 0.0 and 1.0

# For training, we use regular MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Will create loaders per digit in the training function

# For testing our experiment, we'll use degraded MNIST
test_dataset = DegradedMNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True,
    degradation_type=degradation_type,
    severity=severity
)


# Will create loaders per digit in the testing function

# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def add_noise(self, x, noise_factor):
        noisy_x = x + noise_factor * torch.randn_like(x)
        return torch.clamp(noisy_x, 0., 1.)

    def forward(self, x, add_noise=True):
        if add_noise:
            noisy_x = self.add_noise(x, noise_factor)
            z = self.encode(noisy_x)
        else:
            z = self.encode(x)
        return self.decode(z), z


# U-Net based diffusion model for the latent space
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        # Simple MLP for the latent space
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())

        # Concatenate and process through network
        x_t = torch.cat([x, t_emb], dim=1)
        return self.net(x_t)


# Training loop for Denoising Autoencoder
def train_dae(model, dataloader, optimizer, epoch, digit=None):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        reconstructed, latent = model(data)
        loss = F.mse_loss(reconstructed, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 50 == 0:
            digit_str = f" Digit {digit}" if digit is not None else ""
            print(f'DAE Train{digit_str} Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(dataloader)
    digit_str = f" Digit {digit}" if digit is not None else ""
    print(f'====> DAE{digit_str} Epoch: {epoch} Average loss: {avg_loss:.6f}')
    return avg_loss


# Forward diffusion process
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise


# Training loop for Diffusion Model
def train_diffusion(dae_model, diffusion_model, dataloader, optimizer, epoch, digit=None):
    diffusion_model.train()
    dae_model.eval()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        # Get latent representations from DAE
        with torch.no_grad():
            _, latents = dae_model(data, add_noise=False)

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

        if batch_idx % 50 == 0:
            digit_str = f" Digit {digit}" if digit is not None else ""
            print(f'Diffusion Train{digit_str} Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(dataloader)
    digit_str = f" Digit {digit}" if digit is not None else ""
    print(f'====> Diffusion{digit_str} Epoch: {epoch} Average loss: {avg_loss:.6f}')
    return avg_loss


# Sampling (reverse diffusion process)
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
def p_sample_loop(diffusion_model, dae_model, latent_start=None, shape=None, n_samples=8, diffusion_steps_used=None,skip_features=None):
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
    if skip_features is not None:
         decoded_samples = dae_model.decoder(latent_samples, skip_features)
    else:
        decoded_samples = dae_model.decoder(latent_samples)

   

    return decoded_samples, imgs


# Run the degradation experiment
@torch.no_grad()
def run_degradation_experiment(dae_model, diffusion_model, test_loader, digit=None, num_examples=10,
                               diffusion_steps_used=None, save_path="degradation_experiment"):
    import os
    digit_str = f"_digit_{digit}" if digit is not None else ""
    save_path = f"{save_path}{digit_str}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dae_model.eval()
    diffusion_model.eval()

    # Get samples from test set
    all_samples = []
    for (data, original), target in test_loader:
        # data is now a tuple of (degraded, original)
        degraded, original = data,original
        degraded, original = degraded.to(device), original.to(device)

        # Get a few samples for visualization
        if len(all_samples) < num_examples:
            for i in range(min(len(degraded), num_examples - len(all_samples))):
                all_samples.append((degraded[i:i + 1], original[i:i + 1], target[i].item()))
        else:
            break

    # Process each sample
    rows = []
    for i, (degraded, original, label) in enumerate(all_samples):
        # Only process samples of the specified digit if digit is provided
        if digit is not None and label != digit:
            continue

        # Encode the degraded image to get latent
        _, degraded_latent = dae_model(degraded, add_noise=False)

        # Reconstruct from the latent directly
        direct_reconstruction = dae_model.decode(degraded_latent)

        # Apply diffusion to the latent
        denoised_reconstruction, _ = p_sample_loop(
            diffusion_model,
            dae_model,
            latent_start=degraded_latent,
            diffusion_steps_used=diffusion_steps_used
        )

        # Store the results for this sample
        row = {
            'degraded': degraded.cpu(),
            'original': original.cpu(),
            'direct_reconstruction': direct_reconstruction.cpu(),
            'diffusion_reconstruction': denoised_reconstruction.cpu(),
            'label': label
        }
        rows.append(row)

        # Create individual image comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(degraded.cpu().squeeze().numpy(), cmap='gray')
        axes[0].set_title(f'Degraded Input (Digit {label})')
        axes[0].axis('off')

        axes[1].imshow(original.cpu().squeeze().numpy(), cmap='gray')
        axes[1].set_title('Original')
        axes[1].axis('off')

        axes[2].imshow(direct_reconstruction.cpu().squeeze().numpy(), cmap='gray')
        axes[2].set_title('Direct DAE Reconstruction')
        axes[2].axis('off')

        axes[3].imshow(denoised_reconstruction.cpu().squeeze().numpy(), cmap='gray')
        axes[3].set_title('Diffusion Reconstruction')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}/sample_{i}_digit_{label}.png")
        plt.close(fig)

    # Create a grid of all samples if we have any
    if rows:
        fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))

        # Handle case with just one row
        if len(rows) == 1:
            axes = axes.reshape(1, -1)

        for i, row in enumerate(rows):
            axes[i, 0].imshow(row['degraded'].squeeze().numpy(), cmap='gray')
            if i == 0:
                axes[i, 0].set_title('Degraded Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(row['original'].squeeze().numpy(), cmap='gray')
            if i == 0:
                axes[i, 1].set_title('Original')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(row['direct_reconstruction'].squeeze().numpy(), cmap='gray')
            if i == 0:
                axes[i, 2].set_title('Direct DAE Reconstruction')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(row['diffusion_reconstruction'].squeeze().numpy(), cmap='gray')
            if i == 0:
                axes[i, 3].set_title('Diffusion Reconstruction')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}/all_samples_grid.png")
        plt.close(fig)

    return rows


# Function to visualize samples
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


# Train models for a specific digit
def train_digit_models(digit, train_dataset, epochs=epochs):
    print(f"Training models for digit {digit}...")

    # Create digit-specific dataset
    digit_train_dataset = create_digit_dataset(train_dataset, digit)
    digit_train_loader = DataLoader(digit_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    dae = DenoisingAutoencoder(latent_dim).to(device)
    diffusion = DiffusionModel(latent_dim).to(device)

    # Optimizers
    dae_optimizer = optim.Adam(dae.parameters(), lr=lr)
    diffusion_optimizer = optim.Adam(diffusion.parameters(), lr=lr)

    # Train DAE
    print(f"Training DAE for digit {digit}...")
    dae_losses = []
    for epoch in range(1, epochs + 1):
        loss = train_dae(dae, digit_train_loader, dae_optimizer, epoch, digit=digit)
        dae_losses.append(loss)

    # Train diffusion model
    print(f"Training Diffusion for digit {digit}...")
    diffusion_losses = []
    for epoch in range(1, epochs + 1):
        loss = train_diffusion(dae, diffusion, digit_train_loader, diffusion_optimizer, epoch, digit=digit)
        diffusion_losses.append(loss)

        if epoch % 5 == 0:
            print(f"Generating samples for digit {digit} at epoch {epoch}...")
            samples, _ = p_sample_loop(diffusion, dae, shape=(batch_size, latent_dim), n_samples=8)
            visualize_samples(samples, epoch, digit=digit)

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

    # Save models
    os.makedirs('models', exist_ok=True)
    torch.save(dae.state_dict(), f'models/dae_model_digit_{digit}.pth')
    torch.save(diffusion.state_dict(), f'models/diffusion_model_digit_{digit}.pth')

    # Generate final samples
    print(f"Generating final samples for digit {digit}...")
    samples, _ = p_sample_loop(diffusion, dae, shape=(batch_size, latent_dim), n_samples=16)

    # Plot a grid of samples
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(samples):
            img = samples[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"final_samples_digit_{digit}.png")
    plt.close()

    return dae, diffusion


# Run degradation experiment for a specific digit
def run_digit_experiment(digit, dae_model, diffusion_model, test_dataset, degradation_type='rotation', severity=1.0):
    print(f"Running experiment for digit {digit} with {degradation_type} degradation (severity: {severity})...")

    # Create degraded test dataset for this digit
    digit_test_dataset = create_digit_dataset(test_dataset, digit)
    digit_test_loader = DataLoader(digit_test_dataset, batch_size=batch_size, shuffle=False)

    # Run experiment
    experiment_results = run_degradation_experiment(
        dae_model,
        diffusion_model,
        digit_test_loader,
        digit=digit,
        num_examples=10,
        diffusion_steps_used=100,
        save_path=f"degradation_experiment_{degradation_type}_severity_{severity}"
    )

    print(f"Experiment for digit {digit} completed. Results saved.")


# Function for training all digit models
def train_all_digit_models(train_dataset, epochs=10):
    # Train a model for each digit
    digit_models = {}
    for digit in range(10):
        dae, diffusion = train_digit_models(digit, train_dataset, epochs)
        digit_models[digit] = (dae, diffusion)
    return digit_models


# Function to load pre-trained digit models
def load_digit_models(models_dir='models'):
    digit_models = {}
    for digit in range(10):
        dae = DenoisingAutoencoder(latent_dim).to(device)
        diffusion = DiffusionModel(latent_dim).to(device)

        try:
            dae.load_state_dict(torch.load(f'{models_dir}/dae_model_digit_{digit}.pth', map_location=device))
            diffusion.load_state_dict(
                torch.load(f'{models_dir}/diffusion_model_digit_{digit}.pth', map_location=device))
            digit_models[digit] = (dae, diffusion)
            print(f"Loaded models for digit {digit}")
        except FileNotFoundError:
            print(f"Models for digit {digit} not found")

    return digit_models


# Run experiments for all digit models
def run_all_digit_experiments(digit_models, test_dataset, degradation_type='rotation', severity=1.0):
    for digit, (dae, diffusion) in digit_models.items():
        run_digit_experiment(digit, dae, diffusion, test_dataset, degradation_type, severity)


# Main function that ties everything together
def main(train_per_digit=True, digits_to_train=None, run_experiment=True, epochs_per_digit=10):
    # Create regular MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Create degraded test dataset
    test_dataset = DegradedMNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True,
        degradation_type=degradation_type,
        severity=severity
    )

    # Option 1: Train separate models for each digit
    if train_per_digit:
        if digits_to_train is None:
            digits_to_train = range(10)  # Train all digits by default

        digit_models = {}
        for digit in digits_to_train:
            dae, diffusion = train_digit_models(digit, train_dataset, epochs=epochs_per_digit)
            digit_models[digit] = (dae, diffusion)

        if run_experiment:
            run_all_digit_experiments(digit_models, test_dataset, degradation_type, severity)

    # Option 2: Train a single model for all digits (original approach)
    else:
        # Initialize models
        dae = DenoisingAutoencoder(latent_dim).to(device)
        diffusion = DiffusionModel(latent_dim).to(device)

        # Optimizers
        dae_optimizer = optim.Adam(dae.parameters(), lr=lr)
        diffusion_optimizer = optim.Adam(diffusion.parameters(), lr=lr)

        # Create regular data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Train DAE
        print("Training Denoising Autoencoder (all digits)...")
        dae_losses = []
        for epoch in range(1, epochs + 1):
            loss = train_dae(dae, train_loader, dae_optimizer, epoch)
            dae_losses.append(loss)

        # Train diffusion model
        print("Training Diffusion Model (all digits)...")
        diffusion_losses = []
        for epoch in range(1, epochs + 1):
            loss = train_diffusion(dae, diffusion, train_loader, diffusion_optimizer, epoch)
            diffusion_losses.append(loss)

            if epoch % 5 == 0:
                print(f"Generating samples at epoch {epoch}...")
                samples, _ = p_sample_loop(diffusion, dae, shape=(batch_size, latent_dim), n_samples=8)
                visualize_samples(samples, epoch)

        # Save models
        torch.save(dae.state_dict(), 'dae_model_all_digits.pth')
        torch.save(diffusion.state_dict(), 'diffusion_model_all_digits.pth')

        if run_experiment:
            print("Running degradation experiment (all digits)...")
            experiment_results = run_degradation_experiment(
                dae,
                diffusion,
                test_loader,
                num_examples=10,
                diffusion_steps_used=100
            )


if __name__ == "__main__":
    # Train models for all digits with fewer epochs for faster training
    #main(train_per_digit=True, digits_to_train=range(10), epochs_per_digit=40)

    # OR train for specific digits only
    # main(train_per_digit=True, digits_to_train=[0, 1, 2], epochs_per_digit=20)

    # OR train a single model for all digits (original approach)
    # main(train_per_digit=False)

    # OR load pre-trained models and run experiments only
    digit_models = load_digit_models()
    test_dataset = DegradedMNIST(root='./data', train=False, transform=transform, download=True,
                                  degradation_type='noise', severity=0.1)
    run_all_digit_experiments(digit_models, test_dataset, 'noise', 0.1)
