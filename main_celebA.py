import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import torch
import torch.optim as optim
from architectures import FaceAutoencoder120_3skips,FacesAutoencoder,FaceDenoisingAutoencoder120,FaceAutoencoder120_skip,DiffusionModel,MLP,FacesAutoencoder
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import math
from scipy.interpolate import interp1d
from train import train
from losses import loss_L2,loss_L1,loss_ssim,loss_dist_al
from utils import add_noise, add_blur,ldm_reconstruction,reconstruction,calc_metrics
from train import train_diffusion,EarlyStopping,train_all


class SCUTFaceBeautyDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                scores = list(map(float, parts[1:]))
                avg_score = sum(scores) / len(scores)
                self.samples.append((img_name, avg_score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, score = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(score, dtype=torch.float32)

class NoisySCUTFaceDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, noise_level=0.2):
        self.img_dir = img_dir
        self.transform = transform
        self.noise_level = noise_level
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                img_name = line.strip().split()[0]
                self.samples.append(img_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            clean = self.transform(image)
        else:
            clean = TF.to_tensor(image)

        # Add Gaussian noise
        noise = torch.randn_like(clean) * self.noise_level
        noisy = clean + noise
        noisy = torch.clip(noisy, 0., 1.)

        return noisy, clean

def mask_random_patches(image: torch.Tensor, patch_size: int, coverage: float) -> torch.Tensor:
    """
    Apply random black patches to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (int): Size of the square patch.
        coverage (float): Percentage of the image area to cover with patches (0 to 1).

    Returns:
        torch.Tensor: Image tensor with random black patches.
    """
    output = image.clone()
    _, height, width = image.shape
    total_area = height * width
    patch_area = patch_size ** 2
    num_patches = int((coverage * total_area) // patch_area)

    for _ in range(num_patches):
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)

        # Apply black patch
        output[:, top:top + patch_size, left:left + patch_size] = 0

    return output



def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Add salt and pepper noise to a PyTorch tensor image.
    
    Args:
        image (torch.Tensor): Input image tensor (C x H x W) or (B x C x H x W)
        salt_prob (float): Probability of a pixel being replaced with salt (white)
        pepper_prob (float): Probability of a pixel being replaced with pepper (black)
        
    Returns:
        torch.Tensor: Noisy image tensor with same shape as input
    """
    # Create a copy of the image to avoid modifying the original
    noisy_image = image.clone()
    
    # Generate random values between 0 and 1
    noise = torch.rand_like(image)
    
    # Set pixels to white (1.0) where noise < salt_prob
    salt_mask = (noise < salt_prob)
    noisy_image[salt_mask] = 1.0
    
    # Set pixels to black (0.0) where noise > (1 - pepper_prob)
    pepper_mask = (noise > (1 - pepper_prob))
    noisy_image[pepper_mask] = 0.0
    
    return noisy_image


def color_jitter_deformation(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply random color jitter to a PyTorch tensor image.
    
    Args:
        image (torch.Tensor): Input image tensor (C x H x W) or (B x C x H x W)
        brightness (float): Maximum brightness adjustment factor
        contrast (float): Maximum contrast adjustment factor
        saturation (float): Maximum saturation adjustment factor
        hue (float): Maximum hue adjustment factor
        
    Returns:
        torch.Tensor: Color jittered image tensor with same shape as input
    """
    # Make sure input is on CPU for torchvision transforms
    device = image.device
    image_cpu = image.detach().cpu()
    
    # Get original shape to handle both single images and batches
    original_shape = image_cpu.shape
    
    # Handle batch dimension if present
    if len(original_shape) == 4:
        B, C, H, W = original_shape
        # Process each image in the batch
        result = []
        for i in range(B):
            img = image_cpu[i]  # C x H x W
            # Apply random brightness
            if brightness > 0:
                brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
                img = TF.adjust_brightness(img, brightness_factor)
            
            # Apply random contrast
            if contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
                img = TF.adjust_contrast(img, contrast_factor)
            
            # Apply random saturation (only works with 3-channel images)
            if C == 3 and saturation > 0:
                saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
                img = TF.adjust_saturation(img, saturation_factor)
            
            # Apply random hue (only works with 3-channel images)
            if C == 3 and hue > 0:
                hue_factor = random.uniform(-hue, hue)
                img = TF.adjust_hue(img, hue_factor)
                
            result.append(img)
        
        # Stack back into a batch
        deformed_image = torch.stack(result)
    else:
        # No batch dimension, just process single image
        img = image_cpu
        
        # Apply random brightness
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            img = TF.adjust_brightness(img, brightness_factor)
        
        # Apply random contrast
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Apply random saturation (only works with 3-channel images)
        if len(original_shape) == 3 and original_shape[0] == 3 and saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            img = TF.adjust_saturation(img, saturation_factor)
        
        # Apply random hue (only works with 3-channel images)
        if len(original_shape) == 3 and original_shape[0] == 3 and hue > 0:
            hue_factor = random.uniform(-hue, hue)
            img = TF.adjust_hue(img, hue_factor)
            
        deformed_image = img
    
    # Return to original device
    return deformed_image.to(device)

def channel_swap_deformation(image, swap_probability=0.5):
    """
    Randomly swap color channels in an RGB image with a certain probability.
    
    Args:
        image (torch.Tensor): Input image tensor (3 x H x W) or (B x 3 x H x W)
        swap_probability (float): Probability of applying the channel swap
        
    Returns:
        torch.Tensor: Image with possibly swapped channels
    """
    # Check if we should apply the transform
    if random.random() > swap_probability:
        return image
    
    # Create a copy of the image
    result = image.clone()
    
    # Determine original shape
    if len(image.shape) == 4:  # Batch of images
        B, C, H, W = image.shape
        if C != 3:
            return image  # Only works with 3 channel images
        
        # Get a random permutation of [0,1,2]
        permutation = torch.tensor(random.sample([0, 1, 2], 3))
        
        # Apply the permutation to each image in the batch
        for i in range(B):
            result[i] = image[i, permutation]
    
    elif len(image.shape) == 3:  # Single image
        C, H, W = image.shape
        if C != 3:
            return image  # Only works with 3 channel images
        
        # Get a random permutation of [0,1,2]
        permutation = random.sample([0, 1, 2], 3)
        
        # Create new image with swapped channels
        for i, p in enumerate(permutation):
            result[i] = image[p]
    
    return result

import torch
import numpy as np
import random
import math

def add_scribble_deformation(image, num_lines=3, line_width=1, line_color=None, alpha=0.3):
    """
    Add subtle scribble deformation to a PyTorch tensor image using Bézier curves.
    
    Args:
        image (torch.Tensor): Input image tensor (C x H x W) or (B x C x H x W)
        num_lines (int): Number of scribble lines to draw
        line_width (int): Width of the scribble lines in pixels
        line_color (list or None): RGB color of the scribble [r, g, b], each in [0,1] range.
                                   If None, a random color will be used for each line.
        alpha (float): Opacity of the scribbles (0.0 to 1.0)
        
    Returns:
        torch.Tensor: Image with scribble deformation
    """
    # Create a copy of the image to avoid modifying the original
    result = image.clone()
    
    # Handle batch dimension if present
    if len(image.shape) == 4:
        B, C, H, W = image.shape
        for b in range(B):
            result[b] = add_scribble_to_single_image(
                image[b], num_lines, line_width, line_color, alpha, H, W, C)
    else:
        C, H, W = image.shape
        result = add_scribble_to_single_image(
            image, num_lines, line_width, line_color, alpha, H, W, C)
    
    return result

def bezier_curve(p0, p1, p2, num_points=100):
    """
    Generate points on a quadratic Bézier curve.
    
    Args:
        p0 (tuple): Starting point (x0, y0)
        p1 (tuple): Control point (x1, y1)
        p2 (tuple): End point (x2, y2)
        num_points (int): Number of points to generate
        
    Returns:
        tuple: Arrays of x and y coordinates
    """
    t = np.linspace(0, 1, num_points)
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return np.round(x).astype(int), np.round(y).astype(int)

def add_scribble_to_single_image(image, num_lines, line_width, line_color, alpha, H, W, C):
    """Helper function to add scribbles to a single image"""
    # Convert tensor to numpy for easier manipulation
    device = image.device
    result = image.clone().cpu().numpy()
    
    # Create an empty mask for scribbles
    scribble_mask = np.zeros((H, W), dtype=np.float32)
    
    for _ in range(num_lines):
        # Let's create a series of connected Bézier curves
        num_segments = random.randint(1, 3)  # Number of curve segments
        
        # Start point
        start_x = random.randint(0, W-1)
        start_y = random.randint(0, H-1)
        
        last_x, last_y = start_x, start_y
        
        for segment in range(num_segments):
            # Control point - random offset from last point
            max_offset = min(H, W) // 4
            ctrl_x = min(max(0, last_x + random.randint(-max_offset, max_offset)), W-1)
            ctrl_y = min(max(0, last_y + random.randint(-max_offset, max_offset)), H-1)
            
            # End point - random offset from control point
            end_x = min(max(0, ctrl_x + random.randint(-max_offset, max_offset)), W-1)
            end_y = min(max(0, ctrl_y + random.randint(-max_offset, max_offset)), H-1)
            
            # Generate points along the Bézier curve
            x_smooth, y_smooth = bezier_curve(
                (last_x, last_y), 
                (ctrl_x, ctrl_y), 
                (end_x, end_y), 
                num_points=max(30, int(np.hypot(end_x-last_x, end_y-last_y) * 2))
            )
            
            # Clip to valid image coordinates
            x_smooth = np.clip(x_smooth, 0, W-1)
            y_smooth = np.clip(y_smooth, 0, H-1)
            
            # Draw the line segment on the mask
            for i in range(len(x_smooth)):
                # For line width > 1, create a small circle around each point
                for dx in range(-line_width, line_width + 1):
                    for dy in range(-line_width, line_width + 1):
                        x, y = x_smooth[i] + dx, y_smooth[i] + dy
                        if 0 <= x < W and 0 <= y < H:
                            # Apply a soft falloff for line edges
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist <= line_width:
                                intensity = 1.0 - (dist / line_width)
                                scribble_mask[y, x] = max(scribble_mask[y, x], intensity)
            
            # Update start point for the next segment
            last_x, last_y = end_x, end_y
    
    # Choose colors for the scribbles
    for c in range(C):
        if line_color is None:
            # Random color channel value between 0 and 1
            color_val = random.random()
        else:
            color_val = line_color[c % len(line_color)]
        
        # Apply the scribble to the image with alpha blending
        scribble_alpha = scribble_mask * alpha
        result[c] = result[c] * (1 - scribble_alpha) + color_val * scribble_alpha
    
    # Convert back to torch tensor
    return torch.tensor(result, device=device)


def sharpen_image(image, amount=0.5):
    """
    Apply a subtle sharpening filter to a PyTorch tensor image.
    
    Args:
        image (torch.Tensor): Input image tensor (C x H x W) or (B x C x H x W)
        amount (float): Sharpening intensity (0.0 to 2.0 recommended)
                        - 0.0: No sharpening
                        - 0.5: Subtle sharpening
                        - 1.0: Moderate sharpening
                        - 2.0: Strong sharpening
        
    Returns:
        torch.Tensor: Sharpened image tensor with same shape as input
    """
    # Store original device
    device = image.device
    
    # Create a sharpening kernel
    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Handle batch dimension if present
    if len(image.shape) == 4:
        B, C, H, W = image.shape
        
        # Create output tensor
        result = image.clone()
        
        # Process each channel separately
        for b in range(B):
            for c in range(C):
                # Extract single channel and add batch dimension
                img_channel = image[b, c:c+1]
                
                # Apply convolution for Laplacian
                laplacian = F.conv2d(img_channel.unsqueeze(0), kernel, padding=1)
                
                # Add weighted Laplacian to original image for sharpening
                result[b, c] = torch.clamp(image[b, c] + amount * laplacian.squeeze(), 0.0, 1.0)
    else:
        C, H, W = image.shape
        
        # Create output tensor
        result = image.clone()
        
        # Process each channel separately
        for c in range(C):
            # Extract single channel and add batch dimension
            img_channel = image[c:c+1].unsqueeze(0)
            
            # Apply convolution for Laplacian
            laplacian = F.conv2d(img_channel, kernel, padding=1)
            
            # Add weighted Laplacian to original image for sharpening
            result[c] = torch.clamp(image[c] + amount * laplacian.squeeze(), 0.0, 1.0)
    
    return result

DATA_ROOT = './SCUT-FBP5500_v2'
IMG_DIR = os.path.join(DATA_ROOT, 'Images')
TRAIN_LIST = os.path.join(DATA_ROOT, 'train_test_files/split_of_60%training and 40%testing/train.txt')
TEST_LIST = os.path.join(DATA_ROOT, 'train_test_files/split_of_60%training and 40%testing/test.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



if __name__ == "__main__":

   

    seed=42

    retrain_ours = False
    retrain_dae = False
    retrain_diff = False


    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_name='Experiments/Faces_without_eikonal'
    os.makedirs(experiment_name, exist_ok=True)
    model_path = os.path.join(experiment_name,'Models')
    figure_path = os.path.join(experiment_name,'Figures')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    train_diff_cfg = {}
    train_cfg = {}
    eval_cfg={}

    latent_dim = 1024


    train_cfg['lam3'] = 0
    train_cfg['lam6'] = 1
    train_cfg['lam7'] = 1
    train_cfg['min_noise'] = 0.2
    train_cfg['max_noise'] = 0.2
    train_cfg['do_update'] = False
    train_cfg['lr_ae'] = 1e-3
    train_cfg['lr_dnet'] = 1e-5
    train_cfg['latent_dim'] = latent_dim
    train_cfg['batch_size'] = 32
    train_cfg['epochs'] = 2 #20
    train_cfg['mlp_arch']=[200,50,20]#[120,50,20]
    train_cfg['experiment_name'] = experiment_name
    train_cfg['loss_func']=loss_dist_al

    train_diff_cfg['min_noise'] =  train_cfg['min_noise']
    train_diff_cfg['max_noise'] = train_cfg['max_noise']
    train_diff_cfg['dae_epochs'] = 5 #100
    train_diff_cfg['diff_epochs'] = 50 # 50#100
    train_diff_cfg['dae_lr'] = 1e-3 
    train_diff_cfg['diff_lr'] = 1e-3
    train_diff_cfg['latent_dim'] = latent_dim
    train_diff_cfg['loss'] = loss_L1
    train_diff_cfg['diffusion_steps'] = 200

    eval_cfg['method'] = 'dist'
    eval_cfg['n_iters'] = 1
    eval_cfg['fuse_lambda'] =1.0
    eval_cfg['delta_fuse'] = (1.0-eval_cfg['fuse_lambda'])/eval_cfg['n_iters']
    eval_cfg['sigma'] = 10
    



  
    dae_model_path = os.path.join(model_path, f'dae_model_latent{latent_dim}.pth')
    diff_model_path = os.path.join(model_path, f'diff_model_latent{latent_dim}.pth')
    AE_model_path = os.path.join(model_path, f'AE_model_new_latent{latent_dim}.pth')
    #AE_model_path = os.path.join(model_path, f'AE_model_skips3_latent{latent_dim}.pth')
    #AE_model_path = os.path.join(model_path, f'AE_model_latent{latent_dim}.pth')
    

    

    trainset = SCUTFaceBeautyDataset(IMG_DIR, TRAIN_LIST, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
    testset = SCUTFaceBeautyDataset(IMG_DIR, TEST_LIST, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    dae_model = FaceAutoencoder120_skip( bottleneck_size=latent_dim).to(device)
    diff_model = DiffusionModel(latent_dim).to(device)
    AE_model = FaceAutoencoder120_skip( bottleneck_size=latent_dim).to(device)
    #AE_model=FaceAutoencoder120_3skips(bottleneck_size=latent_dim).to(device)
    #AE_model =  FacesAutoencoder(latent_dim=latent_dim).to(device)
    dnet_model = MLP(latent_dim,train_cfg['mlp_arch']).to(device)
   
    if retrain_dae:
        print('start training DAE')
        dae_optimizer = optim.Adam(dae_model.parameters(), lr=train_diff_cfg['dae_lr'])
        dae_losses = train(dae_model, trainloader, train_diff_cfg['loss'], dae_optimizer, device, epochs=train_diff_cfg['dae_epochs'],
                       do_add_noise=True,min_noise=train_diff_cfg['min_noise'], max_noise=train_diff_cfg['max_noise']) 
        torch.save(dae_model.state_dict(), dae_model_path)

    dae_model.load_state_dict(torch.load(dae_model_path,weights_only=True))
    if retrain_diff:
        print(f"Training Diffusion")
        diff_optimizer = optim.Adam(dae_model.parameters(), lr=train_diff_cfg['diff_lr'])
        diffusion_losses = []
        min_loss = float('inf')
        early_stopping = EarlyStopping(patience=15, min_delta=0.00)

        for epoch in range(1, train_diff_cfg['diff_epochs'] + 1):
            loss = train_diffusion(dae_model, diff_model, trainloader, 
                                diff_optimizer, epoch, device, diffusion_steps = train_diff_cfg['diffusion_steps'])

            if early_stopping(loss):
                print(f"Early stopping at epoch {epoch+1}")
                break  # Stop training
            diffusion_losses.append(loss)
            

        print('saving {}'.format(diff_model_path))       
        torch.save(diff_model.state_dict(),diff_model_path)
        
       
    if retrain_ours:
        print('start training our model')
        train_all(train_cfg,AE_model,dnet_model,trainloader,AE_model_path,model_path,device,
                  min_noise=train_cfg['min_noise'],max_noise=train_cfg['max_noise'])    
        
    
    diff_model.load_state_dict(torch.load(diff_model_path,weights_only=True))
    checkpoint = torch.load(AE_model_path,weights_only=True)
    AE_model.load_state_dict(checkpoint['AE_state_dict'])
    dnet_model.load_state_dict(checkpoint['dnet_state_dict'])

    dae_model.eval()
    diff_model.eval()
    AE_model.eval()
    dnet_model.eval()

    all_data = []

    i=0
    for batch in testloader:
        if i >=3:
            break
        inputs, labels = batch
        all_data.append((inputs, labels))
        i+=1

    
    #Optionally, you can concatenate all the batches to get the complete dataset as tensors
    clean = torch.cat([inputs for inputs, _ in all_data])



    #clean,_ = next(iter(testloader))

    #noisy = add_noise(clean, 0.15)

    #noisy = add_blur(clean,kernel=3)
    #noisy = sharpen_image(clean, amount=2.8) #2.8
    #noisy = add_scribble_deformation(clean, num_lines=2, line_width=1, line_color=[0.00,1.000,1.000], alpha=0.7)

    noisy= torch.zeros_like(clean)
    for i in range(clean.shape[0]):
        #noisy[i,:,:,:] = mask_random_patches(clean[i], patch_size=1, coverage=0.01)
        noisy[i,:,:,:] = mask_random_patches(clean[i], patch_size=1, coverage=0.035)
        #noisy[i,:,:,:] = add_salt_and_pepper_noise(clean[i], salt_prob=0.005, pepper_prob=0.005)
        #noisy[i,:,:,:] = color_jitter_deformation(clean[i], brightness=0.02, contrast=0.02, saturation=0.02, hue=0.1)
        #noisy[i,:,:,:]=add_scribble_deformation(noisy[i], num_lines=3, line_width=1, line_color=None, alpha=0.3)
    


    ours_output = reconstruction(noisy.to(device), AE_model, dnet_model, device, 'dist', trainloader, eval_cfg=eval_cfg)

    ours_mse,ours_psnr,ours_ssim = calc_metrics(ours_output,clean)


   
    with torch.no_grad():
        dae_output = dae_model(noisy.to(device)).cpu()
        diff_output = ldm_reconstruction(noisy.to(device), dae_model, diff_model, device, diffusion_steps=train_diff_cfg['diffusion_steps'] )
        

    dae_mse,dae_psnr,dae_ssim = calc_metrics(dae_output,clean)
    diff_mse,diff_psnr,diff_ssim = calc_metrics(dae_output,clean)
    print(f"Ours Mean SSIM: {ours_ssim:.3f} {ours_psnr:.3f} {ours_mse:.3f}")
    print(f"dae Mean SSIM: {dae_ssim:.3f} {dae_psnr:.3f} {dae_mse:.3f}")
    print(f"diff Mean SSIM: {diff_ssim:.3f} {diff_psnr:.3f} {diff_mse:.3f}")
    # Show original, noisy, and denoised
    fig, axs = plt.subplots(4, 5, figsize=(10, 6))
    index=0
    for i in range(5):
        #axs[0, i].imshow(clean[i+index].permute(1, 2, 0))
        #axs[0, i].set_title("Clean")
        axs[0, i].imshow(noisy[i+index].permute(1, 2, 0))
        #axs[1, i].set_title("Noisy")
        #axs[2, i].imshow(dae_output[i+index].permute(1, 2, 0))
        #axs[2, i].set_title("DAE Denoised")
        axs[1, i].imshow(diff_output[i+index].permute(1, 2, 0))
        #axs[3, i].set_title("Diffusion Denoised")
        axs[2, i].imshow(ours_output[i+index].permute(1, 2, 0))
        #axs[4, i].set_title("Ours Denoised")
        axs[3, i].imshow(clean[i+index].permute(1, 2, 0))
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout()


    fig, axs = plt.subplots(3, 4)
    index=2
    for i in range(3):
        
        axs[i, 0].imshow(noisy[i+index].permute(1, 2, 0))
       
        axs[i, 1].imshow(diff_output[i+index].permute(1, 2, 0))
      
        axs[i, 2].imshow(ours_output[i+index].permute(1, 2, 0))
       
        axs[i, 3].imshow(clean[i+index].permute(1, 2, 0))
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout()



    plt.savefig(os.path.join(figure_path, 'denoised.png')) 
    #plt.show() 


######
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid


# class DenoisingAutoencoder(nn.Module):
#     def __init__(self, bottleneck_size=1024):
#         super(DenoisingAutoencoder, self).__init__()
        
#         # Define encoder blocks individually to access intermediate outputs for skip connections
#         # First encoder block
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 120x120 -> 60x60
        
#         # Second encoder block
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 60x60 -> 30x30
        
#         # Third encoder block
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 30x30 -> 15x15
        
#         # Fourth encoder block
#         self.enc4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 15x15 -> 7x7 (approx)
        
#         # Flattened bottleneck
#         self.flatten = nn.Flatten()
        
#         # Calculate size after encoding convolutional layers
#         # With 120x120 input, after 4 MaxPool layers (dividing by 2^4 = 16)
#         # We get 7.5x7.5 which becomes 7x7
#         self.feature_size = 256 * 7 * 7
        
#         # Dense bottleneck layer - LARGER bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Linear(self.feature_size, bottleneck_size),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Decoder begins with dense layer back to pre-flatten size
#         self.decoder_dense = nn.Sequential(
#             nn.Linear(bottleneck_size, self.feature_size),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Define decoder blocks with transposed convolutions for better upsampling
#         # First decoder block
#         self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.dec4 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 512 due to skip connection
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Second decoder block
#         self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256 due to skip connection
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Third decoder block
#         self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128 due to skip connection
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Fourth decoder block
#         self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 64 due to skip connection
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Additional handling for odd dimensions
#         self.final_upconv = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=True)
        
#         # Final output layer
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # Output in [0, 1] range
#         )

#     def encode(self, x):
#         # Store intermediate outputs for skip connections
#         e1 = self.enc1(x)
#         p1 = self.pool1(e1)
        
#         e2 = self.enc2(p1)
#         p2 = self.pool2(e2)
        
#         e3 = self.enc3(p2)
#         p3 = self.pool3(e3)
        
#         e4 = self.enc4(p3)
#         p4 = self.pool4(e4)
        
#         flat = self.flatten(p4)
#         bottleneck = self.bottleneck(flat)
        
#         # Store intermediate features for decoder
#         return bottleneck, (e1, e2, e3, e4)
    
#     def decode(self, x, skip_features):
#         e1, e2, e3, e4 = skip_features
        
#         x = self.decoder_dense(x)
#         x = x.view(-1, 256, 7, 7)  # Reshape to match encoder output dimensions
        
#         # Use skip connections in decoder
#         x = self.upconv4(x)
#         # Handle potential size mismatch
#         if x.size() != e4.size():
#             x = nn.functional.interpolate(x, size=e4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x, e4], dim=1)  # Skip connection
#         x = self.dec4(x)
        
#         x = self.upconv3(x)
#         if x.size() != e3.size():
#             x = nn.functional.interpolate(x, size=e3.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x, e3], dim=1)  # Skip connection
#         x = self.dec3(x)
        
#         x = self.upconv2(x)
#         if x.size() != e2.size():
#             x = nn.functional.interpolate(x, size=e2.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x, e2], dim=1)  # Skip connection
#         x = self.dec2(x)
        
#         x = self.upconv1(x)
#         if x.size() != e1.size():
#             x = nn.functional.interpolate(x, size=e1.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x, e1], dim=1)  # Skip connection
#         x = self.dec1(x)
        
#         # Make sure we get back to 120x120
#         if x.size()[2:] != (120, 120):
#             x = self.final_upconv(x)
        
#         # Final convolution
#         x = self.final_conv(x)
#         return x
    
#     def forward(self, x):
#         # Encode
#         encoded, skip_features = self.encode(x)
        
#         # Decode with skip connections
#         decoded = self.decode(encoded, skip_features)
        
#         return decoded


# class SCUT_FBP5500_Dataset(Dataset):
#     def __init__(self, root_dir, transform=None, noise_factor=0.1, augment=True):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.noise_factor = noise_factor
#         self.augment = augment
#         self.image_files = [f for f in os.listdir(root_dir) 
#                             if os.path.isfile(os.path.join(root_dir, f)) and 
#                             (f.endswith('.jpg') or f.endswith('.png'))]
        
#     def __len__(self):
#         return len(self.image_files)
    
#     def add_realistic_noise(self, image):
#         """Add more realistic noise patterns than just Gaussian noise"""
#         # Base noise - reduced factor for less blurriness
#         noise = torch.randn_like(image) * self.noise_factor
        
#         # Add salt and pepper noise occasionally
#         if torch.rand(1).item() > 0.7:
#             salt_pepper_mask = torch.rand_like(image)
#             salt_mask = salt_pepper_mask > 0.99  # 1% salt noise
#             pepper_mask = salt_pepper_mask < 0.01  # 1% pepper noise
            
#             # Apply salt (white) and pepper (black) noise
#             image = image * (~salt_mask).float() + salt_mask.float()
#             image = image * (~pepper_mask).float()
        
#         # Add JPEG compression artifacts occasionally
#         if torch.rand(1).item() > 0.7:
#             # Convert to numpy, apply JPEG compression simulation
#             img_np = image.permute(1, 2, 0).cpu().numpy()
#             img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
#             # Save with low quality to simulate compression artifacts
#             buffer = BytesIO()
#             quality = np.random.randint(30, 70)  # Random low-mid quality
#             img_pil.save(buffer, format="JPEG", quality=quality)
#             buffer.seek(0)
            
#             # Load back the compressed image
#             img_compressed = Image.open(buffer)
#             img_np = np.array(img_compressed).astype(np.float32) / 255.0
            
#             # Convert back to tensor
#             image = torch.from_numpy(img_np).permute(2, 0, 1)
        
#         # Add regular Gaussian noise
#         noisy_image = image + noise
        
#         # Add motion blur occasionally
#         if torch.rand(1).item() > 0.8:
#             kernel_size = np.random.choice([3, 5, 7])
#             kernel = torch.zeros((kernel_size, kernel_size))
            
#             # Create a line in a random direction
#             direction = np.random.randint(4)
#             if direction == 0:  # Horizontal
#                 kernel[kernel_size // 2, :] = 1.0 / kernel_size
#             elif direction == 1:  # Vertical
#                 kernel[:, kernel_size // 2] = 1.0 / kernel_size
#             elif direction == 2:  # Diagonal \
#                 for i in range(kernel_size):
#                     kernel[i, i] = 1.0 / kernel_size
#             else:  # Diagonal /
#                 for i in range(kernel_size):
#                     kernel[i, kernel_size - 1 - i] = 1.0 / kernel_size
                    
#             # Apply the kernel to each channel
#             kernel = kernel.to(image.device)
#             for c in range(image.shape[0]):
#                 channel = image[c:c+1].unsqueeze(0)  # Add batch dimension
#                 blurred = nn.functional.conv2d(
#                     channel, 
#                     kernel.unsqueeze(0).unsqueeze(0), 
#                     padding=kernel_size//2
#                 )
#                 noisy_image[c] = blurred.squeeze()
                
#         return torch.clamp(noisy_image, 0., 1.)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         # Data augmentation for training
#         if self.augment:
#             # Random horizontal flip
#             if torch.rand(1).item() > 0.5:
#                 image = torch.flip(image, [2])  # Flip horizontally
                
#             # Random slight rotation (up to 10 degrees)
#             if torch.rand(1).item() > 0.7:
#                 angle = torch.rand(1).item() * 20 - 10  # -10 to 10 degrees
#                 image = transforms.functional.rotate(image, angle)
                
#             # Random slight color jitter
#             if torch.rand(1).item() > 0.7:
#                 brightness = 0.05 * torch.rand(1).item() + 0.95  # 0.95-1.0
#                 contrast = 0.1 * torch.rand(1).item() + 0.95  # 0.95-1.05
#                 saturation = 0.1 * torch.rand(1).item() + 0.95  # 0.95-1.05
                
#                 # Apply color jitter
#                 if brightness != 1:
#                     image = transforms.functional.adjust_brightness(image, brightness)
#                 if contrast != 1:
#                     image = transforms.functional.adjust_contrast(image, contrast)
#                 if saturation != 1:
#                     image = transforms.functional.adjust_saturation(image, saturation)
        
#         # Create noisy version with more realistic noise
#         noisy_image = self.add_realistic_noise(image)
        
#         # Return both the original and noisy version
#         return noisy_image, image
        
#     def get_statistics(self):
#         """Calculate dataset mean and std for normalization"""
#         # This is useful for normalizing the dataset properly
#         mean = torch.zeros(3)
#         std = torch.zeros(3)
#         for img_name in self.image_files[:100]:  # Use a subset for speed
#             img_path = os.path.join(self.root_dir, img_name)
#             img = Image.open(img_path).convert('RGB')
#             img_tensor = transforms.ToTensor()(img)
#             mean += img_tensor.mean(dim=[1, 2])
#             std += img_tensor.std(dim=[1, 2])
            
#         mean /= min(len(self.image_files), 100)
#         std /= min(len(self.image_files), 100)
#         return mean, std


# def add_noise(image, noise_factor=0.2):
#     """Add random noise to an image tensor."""
#     noise = torch.randn_like(image) * noise_factor
#     noisy_image = image + noise
#     return torch.clamp(noisy_image, 0., 1.)


# def train_autoencoder(model, train_loader, num_epochs=10, lr=0.001, device='cuda'):
#     """Train the autoencoder with combined loss functions for sharper reconstructions."""
#     model = model.to(device)
    
#     # MSE loss - basic reconstruction loss
#     mse_criterion = nn.MSELoss()
    
#     # SSIM loss - structural similarity for better perceptual quality
#     try:
#         import pytorch_msssim
#         ssim_criterion = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=3)
#     except ImportError:
#         print("pytorch_msssim not found, using only MSE loss. Install with 'pip install pytorch-msssim' for better results.")
#         ssim_criterion = None
    
#     # For edge preservation - Sobel filter approximation
#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
#     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
#     # Adam optimizer with slightly lower learning rate for stability
#     optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
#     # Learning rate scheduler for better convergence
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
        
#         for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
#             noisy_imgs = noisy_imgs.to(device)
#             clean_imgs = clean_imgs.to(device)
            
#             # Forward pass
#             outputs = model(noisy_imgs)
            
#             # Basic reconstruction loss
#             mse_loss = mse_criterion(outputs, clean_imgs)
            
#             # Edge preservation loss (using Sobel filters)
#             # Convert to grayscale for edge detection
#             clean_gray = 0.299 * clean_imgs[:, 0:1] + 0.587 * clean_imgs[:, 1:2] + 0.114 * clean_imgs[:, 2:3]
#             output_gray = 0.299 * outputs[:, 0:1] + 0.587 * outputs[:, 1:2] + 0.114 * outputs[:, 2:3]
            
#             # Apply Sobel filters
#             clean_edges_x = nn.functional.conv2d(clean_gray, sobel_x, padding=1)
#             clean_edges_y = nn.functional.conv2d(clean_gray, sobel_y, padding=1)
#             clean_edges = torch.sqrt(clean_edges_x**2 + clean_edges_y**2 + 1e-8)
            
#             output_edges_x = nn.functional.conv2d(output_gray, sobel_x, padding=1)
#             output_edges_y = nn.functional.conv2d(output_gray, sobel_y, padding=1)
#             output_edges = torch.sqrt(output_edges_x**2 + output_edges_y**2 + 1e-8)
            
#             edge_loss = mse_criterion(output_edges, clean_edges)
            
#             # SSIM loss if available (structural similarity)
#             if ssim_criterion:
#                 ssim_loss = 1 - ssim_criterion(outputs, clean_imgs)
#                 # Combined loss - balance between all components
#                 loss = 0.6 * mse_loss + 0.2 * edge_loss + 0.2 * ssim_loss
#             else:
#                 # Without SSIM, just use MSE and edge preservation
#                 loss = 0.7 * mse_loss + 0.3 * edge_loss
            
#             # Backward pass and optimize
#             optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if (batch_idx + 1) % 10 == 0:
#                 loss_info = f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
#                 loss_info += f'Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Edge: {edge_loss.item():.4f}'
#                 if ssim_criterion:
#                     loss_info += f', SSIM: {ssim_loss.item():.4f}'
#                 print(loss_info)
        
#         avg_loss = total_loss / len(train_loader)
#         print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
#         # Update learning rate based on validation loss
#         scheduler.step(avg_loss)
    
#     return model


# def visualize_reconstructions(model, data_loader, num_examples=5, device='cuda'):
#     """Visualize original images, noisy versions, and reconstructions."""
#     model.eval()
#     with torch.no_grad():
#         for noisy_imgs, clean_imgs in data_loader:
#             noisy_imgs = noisy_imgs.to(device)
#             clean_imgs = clean_imgs.to(device)
            
#             # Forward pass
#             reconstructions = model(noisy_imgs)
            
#             # Move to CPU for visualization
#             clean_imgs = clean_imgs.cpu()
#             noisy_imgs = noisy_imgs.cpu()
#             reconstructions = reconstructions.cpu()
            
#             # Plot examples
#             fig, axes = plt.subplots(3, num_examples, figsize=(15, 6))
            
#             for i in range(num_examples):
#                 if i >= len(clean_imgs):
#                     break
                    
#                 # Original
#                 axes[0, i].imshow(clean_imgs[i].permute(1, 2, 0))
#                 axes[0, i].set_title('Original')
#                 axes[0, i].axis('off')
                
#                 # Noisy
#                 axes[1, i].imshow(noisy_imgs[i].permute(1, 2, 0))
#                 axes[1, i].set_title('Noisy')
#                 axes[1, i].axis('off')
                
#                 # Reconstruction
#                 axes[2, i].imshow(reconstructions[i].permute(1, 2, 0))
#                 axes[2, i].set_title('Reconstructed')
#                 axes[2, i].axis('off')
            
#             plt.tight_layout()
#             plt.show()
#             break  # Just show the first batch


# def main():
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize((120, 120)),
#         transforms.ToTensor(),
#     ])
    
#     # Create dataset and dataloader
#     # Replace with your actual path to the SCUT-FBP5500 dataset
#     dataset = SCUT_FBP5500_Dataset(
#         root_dir='path/to/SCUT-FBP5500/Images',
#         transform=transform,
#         noise_factor=0.2
#     )
    
#     # Create data loaders
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=32, 
#         shuffle=True, 
#         num_workers=4
#     )
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=32, 
#         shuffle=False, 
#         num_workers=4
#     )
    
#     # Create the model
#     model = DenoisingAutoencoder()
#     print(model)
    
#     # Train the model
#     model = train_autoencoder(model, train_loader, num_epochs=20, lr=0.001, device=device)
    
#     # Save the trained model
#     torch.save(model.state_dict(), 'denoising_autoencoder_scut_fbp5500.pth')
    
#     # Visualize some reconstructions
#     visualize_reconstructions(model, test_loader, num_examples=5, device=device)
    
#     # Extract and visualize the bottleneck features for a batch
#     model.eval()
#     with torch.no_grad():
#         for noisy_imgs, _ in test_loader:
#             noisy_imgs = noisy_imgs.to(device)
#             bottleneck_features = model.encode(noisy_imgs)
#             print(f"Bottleneck features shape: {bottleneck_features.shape}")
            
#             # You can analyze these bottleneck features for clustering, visualization, etc.
#             break


# if __name__ == "__main__":
#     main()

    