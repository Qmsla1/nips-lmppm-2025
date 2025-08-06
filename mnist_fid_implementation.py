import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from utils import generate_gaussian_noise_images, to_np, reconstruction


# === STEP 1: Feature extractor for MNIST ===
class MNISTFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 256),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(256, feature_dim)  # Final feature dimension
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.features(x)


def load_or_train_feature_extractor(device, train_loader=None, epochs=5, feature_dim=128):
    """Load pretrained feature extractor or train a simple one"""
    model = MNISTFeatureExtractor(feature_dim=feature_dim).to(device)
    model_path = 'mnist_feature_extractor.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained feature extractor")
    elif train_loader is not None:
        print("Training feature extractor...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                features = model.features(data)
                output = model.classifier(features)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        torch.save(model.state_dict(), model_path)
        print("Saved trained feature extractor")

    model.eval()
    return model


# === STEP 2: Feature extraction ===
def get_features(data_loader, model, device, max_samples=None):
    """Extract features from data loader"""
    features = []
    sample_count = 0

    with torch.no_grad():
        for x, _ in data_loader:
            if max_samples and sample_count >= max_samples:
                break

            x = x.to(device)
            f = model(x)
            features.append(f.cpu().numpy())
            sample_count += len(x)

            # Process in smaller chunks to avoid memory issues
            if len(features) > 20:  # Process every 20 batches
                break

    return np.concatenate(features, axis=0)


def get_features_from_images(images, model, device, batch_size=128):
    """Extract features from tensor of images"""
    if isinstance(images, torch.Tensor):
        images = images.cpu()

    dataset = TensorDataset(images, torch.zeros(len(images)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return get_features(loader, model, device)


# === STEP 3: FID computation ===
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Distance between two Gaussians"""
    # Add small epsilon to diagonal for numerical stability
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    try:
        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception as e:
        print(f"Warning: sqrtm failed, using approximation. Error: {e}")
        # Use eigenvalue decomposition as fallback
        try:
            # Alternative calculation using eigenvalues
            product = sigma1 @ sigma2
            eigenvals, eigenvecs = np.linalg.eigh(product)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            covmean = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        except:
            # Last resort: use identity approximation
            covmean = np.sqrt(np.trace(sigma1) * np.trace(sigma2)) * np.eye(sigma1.shape[0])

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_fid_from_loaders(real_loader, fake_loader, model, device):
    """Compute FID from two data loaders"""
    real_features = get_features(real_loader, model, device)
    fake_features = get_features(fake_loader, model, device)

    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    mean_diff = mu2 - mu1
    eigvals = np.linalg.eigvalsh(sigma2)
    return fid_value, mean_diff, eigvals


def compute_fid_from_images(real_images, fake_images, model, device):
    """Compute FID from image tensors"""
    real_features = get_features_from_images(real_images, model, device)
    fake_features = get_features_from_images(fake_images, model, device)

    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    mean_diff = mu2 - mu1
    eigvals = np.linalg.eigvalsh(sigma2)
    return fid_value, mean_diff, eigvals


def save_debug_images(real_images, generated_images, digit):
    count = min(8, len(real_images))
    fig, axes = plt.subplots(2, count, figsize=(2 * count, 4))
    for i in range(count):
        axes[0, i].imshow(real_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(generated_images[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.tight_layout()
    path = f'fid_debug_digit_{digit}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# === STEP 4: Generate images from noise using your models ===
def generate_images_from_noise(digit_list, model_AE_list, model_dnet_list, train_loader_list,
                               device, num_samples_per_digit=1000, **kwargs):
    """Generate images from Gaussian noise using your trained models"""
    all_generated_images = []
    all_labels = []

    for digit in digit_list:
        print(f"Generating {num_samples_per_digit} images for digit {digit}")

        model = model_AE_list[digit]
        dnet = model_dnet_list[digit]
        train_loader = train_loader_list[digit]

        model.eval()
        dnet.eval()

        # Get a reference image for shape
        test_images, _ = next(iter(train_loader))
        test_images = test_images.to(device)

        generated_images_for_digit = []
        samples_generated = 0

        while samples_generated < num_samples_per_digit:
            # Generate images in manageable batches until we reach the requested
            # number of samples for this digit. ``num_samples_per_digit`` controls
            # how many images we want in total and we subtract the amount already
            # produced to avoid overshooting the target.
            batch_size = min(64, num_samples_per_digit - samples_generated)

            # Generate pure Gaussian noise
            gaussian_noise = torch.randn(batch_size, 1, 28, 28, device=device) * 0.5 + 0.5
            gaussian_noise = torch.clamp(gaussian_noise, 0, 1)

            # Use your reconstruction method to convert noise to digit
            generated_batch = reconstruction(gaussian_noise, model, dnet, device,
                                             kwargs.get('method', 'dist'), train_loader,
                                             eval_cfg=kwargs.get('eval_cfg'))

            generated_images_for_digit.append(generated_batch.cpu())
            samples_generated += batch_size

        # Concatenate all generated images for this digit
        digit_images = torch.cat(generated_images_for_digit, dim=0)[:num_samples_per_digit]
        all_generated_images.append(digit_images)
        all_labels.extend([digit] * num_samples_per_digit)

    # Concatenate all generated images
    all_generated_images = torch.cat(all_generated_images, dim=0)
    all_labels = torch.tensor(all_labels)

    return all_generated_images, all_labels


# === STEP 5: Main FID evaluation function ===
def evaluate_fid_per_digit(digit_list, model_AE_list, model_dnet_list, train_loader_list,
                           device, num_samples_per_digit=1000, **kwargs):
    """Evaluate FID for each digit separately"""

    # Load feature extractor with smaller feature dimension
    first_digit = digit_list[0]
    feature_extractor = load_or_train_feature_extractor(device, train_loader_list[first_digit],
                                                        feature_dim=128)

    fid_scores = {}

    for digit in digit_list:
        print(f"\nEvaluating FID for digit {digit}")

        # Get real images for this digit (reduce number of samples)
        real_images = []
        real_labels = []
        target_samples = min(num_samples_per_digit, 500)  # Limit to 500 samples
        for images, labels in train_loader_list[digit]:
            real_images.append(images)
            real_labels.append(labels)
            if len(torch.cat(real_images)) >= target_samples:
                break

        real_images = torch.cat(real_images)[:target_samples]

        # Generate fake images for this digit
        model = model_AE_list[digit]
        dnet = model_dnet_list[digit]
        train_loader = train_loader_list[digit]

        model.eval()
        dnet.eval()

        generated_images = []
        samples_generated = 0
        target_samples = min(num_samples_per_digit, 500)  # Match real images

        while samples_generated < target_samples:
            batch_size = min(128, num_samples_per_digit - samples_generated)

            # Generate pure Gaussian noise
            gaussian_noise = torch.randn(batch_size, 1, 28, 28, device=device) * 0.5 + 0.5
            gaussian_noise = torch.clamp(gaussian_noise, 0, 1)

            # Generate images from noise
            generated_batch = reconstruction(gaussian_noise, model, dnet, device,
                                             kwargs.get('method', 'dist'), train_loader,
                                             eval_cfg=kwargs.get('eval_cfg'))

            generated_images.append(generated_batch.cpu())
            samples_generated += batch_size

        generated_images = torch.cat(generated_images)[:target_samples]

        # Save a small sample of real vs. generated images for quick visual
        # inspection. This does **not** limit the number of images used for the
        # FID computation, which relies on the full ``target_samples`` set above.
        debug_path = save_debug_images(real_images[:14], generated_images[:14], digit)
        print(f"Saved debug images to: {debug_path}")

        # Compute FID and additional stats for this digit
        fid_score, mean_diff, eigvals = compute_fid_from_images(real_images, generated_images,
                                            feature_extractor, device)
        fid_scores[digit] = fid_score

        print(f"FID score for digit {digit}: {fid_score:.2f}")
        print(f"Mean feature difference (L2): {np.linalg.norm(mean_diff):.4f}")
        print(f"Eigenvalues of generated covariance: {eigvals}")

    # Compute overall FID
    overall_fid = np.mean(list(fid_scores.values()))
    print(f"\nOverall FID score: {overall_fid:.2f}")

    return fid_scores, overall_fid


def evaluate_fid_all_digits(digit_list, model_AE_list, model_dnet_list, train_loader_list,
                            device, num_samples_total=10000, **kwargs):
    """Evaluate FID for all digits together"""

    # Load feature extractor
    first_digit = digit_list[0]
    feature_extractor = load_or_train_feature_extractor(device, train_loader_list[first_digit])

    print(f"Generating {num_samples_total} images from noise...")

    # Generate images from all digit models
    generated_images, generated_labels = generate_images_from_noise(
        digit_list, model_AE_list, model_dnet_list, train_loader_list, device,
        num_samples_per_digit=num_samples_total // len(digit_list), **kwargs
    )

    # Get real training images
    print("Loading real training images...")
    real_images = []
    real_labels = []

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Sample from training set
    indices = torch.randperm(len(train_dataset))[:num_samples_total]
    for idx in indices:
        img, label = train_dataset[idx]
        real_images.append(img)
        real_labels.append(label)

    real_images = torch.stack(real_images)
    real_labels = torch.tensor(real_labels)

    # Compute FID
    fid_score, _, _ = compute_fid_from_images(real_images, generated_images,
                                        feature_extractor, device)

    print(f"Overall FID score: {fid_score:.2f}")
    return fid_score


# === Integration with your existing code ===
def run_fid_evaluation(digit_list, model_AE_list, model_dnet_list, train_loader_list,
                       device, eval_cfg, per_digit=True):
    """Main function to run FID evaluation"""

    print("Starting FID evaluation...")

    if per_digit:
        fid_scores, overall_fid = evaluate_fid_per_digit(
            digit_list, model_AE_list, model_dnet_list, train_loader_list,
            device, num_samples_per_digit=1000,
            eval_cfg=eval_cfg, method=eval_cfg['method']
        )
        return fid_scores, overall_fid
    else:
        overall_fid = evaluate_fid_all_digits(
            digit_list, model_AE_list, model_dnet_list, train_loader_list,
            device, num_samples_total=10000,
            eval_cfg=eval_cfg, method=eval_cfg['method']
        )
        return {}, overall_fid