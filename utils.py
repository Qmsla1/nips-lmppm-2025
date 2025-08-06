import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ldm import p_sample_loop
from prettytable import PrettyTable

def grad_d_x(x, d):
    grad_d = torch.autograd.grad(d, x, grad_outputs=torch.ones_like(d), create_graph=True)[0]
    grad_d_norm = torch.norm(grad_d, p=2, dim=(1, 2, 3))
    return grad_d, grad_d_norm

def grad_d_z(z, d, eps=0.0):
    grad_d = torch.autograd.grad(d, z, grad_outputs=torch.ones_like(d), create_graph=True)[0]
    #grad_d_norm = torch.norm(grad_d+eps, p=2, dim=(1))
    grad_d_norm =torch.sqrt(torch.sum(grad_d**2, dim=1) + eps)
    return grad_d, grad_d_norm

def to_tens(A,device='cuda'):
    X = torch.from_numpy(A).to(device)
    X = X.float()
    return X

def to_np(A):
    return A.cpu().detach().numpy()

def add_noise(images, noise_factor=0.4):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

def add_blur(images, kernel=3):
    blurred_img = transforms.functional.gaussian_blur(images,kernel)
    blurred_img = torch.clamp(blurred_img, 0., 1.)
    return blurred_img

def add_rotate(img, min_angle=-50, max_angle=50):
    angle = random.uniform(min_angle, max_angle)
    rotated_img = transforms.functional.rotate(img,angle,InterpolationMode.BILINEAR)
    rotated_img = torch.clamp(rotated_img, 0., 1.)
    return rotated_img

def add_elastic_transform(img, alpha=1.0, sigma=50.0):
    elastic_transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma, interpolation=2)
    elastic_img = elastic_transform(img)
    return elastic_img

def add_down_sample(img, scale_factor=0.5):
    height,width=img.shape[-1],img.shape[-2]
    #height, width = img.shape[1], img.shape[2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    down_sample = transforms.Resize((new_height,new_width),InterpolationMode.BILINEAR)
    up_sample = transforms.Resize((height,width),InterpolationMode.BILINEAR)
    down_sample_img = down_sample(img)
    up_sample_img = up_sample(down_sample_img)
    return up_sample_img

def calc_mean_cov(train_loader, model):
    
    with torch.no_grad():

        train_images, _ = next(iter(train_loader))
        z_clean = model.encoder(train_images).detach().numpy()
        z_mean = np.mean(z_clean,axis=0)
        z_cov=np.cov(z_clean,rowvar=False)

        inv_cov = np.linalg.pinv(z_cov)

    prob_params = {'z_mean': z_mean, 'inv_cov': inv_cov, 'z_cov': z_cov}    
    return prob_params


def calc_mahalanobis(z, z_mean, inv_cov):
    """
    Compute the multivariate Gaussian probability density for a batch of samples.

    Parameters:
    z : np.ndarray (B, d) -> Batch of d-dimensional samples
    z_mean : np.ndarray (d,) -> Mean vector of the distribution
    inv_cov : np.ndarray (d, d) -> Inverse covariance matrix (precision matrix)

    Returns:
    np.ndarray (B,) -> Probabilities for each sample
    """
    # Ensure proper shapes
    z = np.atleast_2d(z)  # Ensure z is at least 2D (B, d)
    
    # Compute difference
    diff = z - z_mean  # (B, d) - (d,) -> (B, d)

    # Compute quadratic form: (B, d) @ (d, d) @ (d, B) -> (B,)
    mahalanobis = np.einsum('bi,ij,bj->b', diff, inv_cov, diff)
    
    # Compute probability
    #prob = np.exp(exponent)
    
    return mahalanobis  # Shape (B,)
def calc_probability_old(z, z_mean, z_cov, inv_cov):
    #partition_function = np.sqrt((2 * np.pi) ** len(z) * np.linalg.det(z_cov))
    z = z - z_mean
    prob = np.exp(-0.5 * np.dot(z, np.dot(inv_cov, z.T)))
    return prob
def generate_images_from_prob(model,z_mean, z_cov, inv_cov, device='cpu',n=80):

    model.eval()
    with torch.no_grad():
        rng = np.random.default_rng(seed=42)
        prob_list = []
        samples = rng.multivariate_normal(z_mean, z_cov, n)
        for z in samples:
            prob=calc_probability(z, z_mean, z_cov, inv_cov)
            prob_list.append(f'{prob:.1e}')

        samples = to_tens(samples,device=device)
        new_images = model.decoder(samples).squeeze().detach().numpy()
        plot_multi_images(new_images,prob_list,figs_in_line=10)

def plot_multi_images(images,captions=None,figs_in_line=20,path=None,edge_color=None):
    N = images.shape[0]
    fig=plt.figure(figsize=(figs_in_line,N//figs_in_line))
    for ii in range(N):
        plt.subplot(int(np.ceil(N / figs_in_line)), figs_in_line, ii + 1)
        plt.imshow(images[ii,:,:], cmap='gray')
        if captions is not None:
            plt.title(captions[ii])
        plt.axis('off')
    plt.tight_layout()
    if edge_color is not None:
        fig.patch.set_linewidth(10)  # Set the line width
        fig.patch.set_edgecolor(edge_color) 
    #fig.patch.set_edgecolor('darkblue') 
    if path is not None:
        plt.savefig(path)
    plt.close(fig)

def calc_z_components(images, model, dnet, device, method, train_loader, **kwargs):

    eval_cfg = kwargs.get('eval_cfg', None)
    if method == 'L2':
        with torch.no_grad():
            z_batch = model.encoder(images)
            return z_batch, None, None
    

    sigma=eval_cfg['sigma']
    train_images, _ = next(iter(train_loader))
    train_images = train_images.to(device)
    model.eval()
    dnet.eval()
    latents = model.encoder(train_images)
    z_clean = latents[0] if isinstance(latents, tuple) else latents
    #z_clean = model.encoder(train_images)
    latents = model.encoder(images)
    z_batch = latents[0] if isinstance(latents, tuple) else latents
    #z_batch = model.encoder(images)

   
    z_bar, z_shift, d_val = update_z(z_batch, dnet, z_clean, sigma, device)


    return z_batch, z_bar, z_shift, d_val


def update_z(z_batch, dnet, z_clean, sigma, device):
    z_batch_np = to_np(z_batch)
    z_clean_np = to_np(z_clean)
    z_batch_reshaped = z_batch_np.reshape(z_batch_np.shape[0], 1, -1)
    z_clean_reshaped = z_clean_np.reshape(1, z_clean_np.shape[0], -1)

    # Calculate differences
    differences = -np.abs(z_batch_reshaped - z_clean_reshaped) ** 1 / sigma

    # Apply numerical stability: clip differences to prevent underflow
    differences = np.clip(differences, -50, 0)  # Prevent exp() underflow

    exp_terms = np.exp(differences)

    # Add small epsilon to prevent division by zero
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    eps = 1e-12
    sum_exp = np.maximum(sum_exp, eps)  # Ensure sum is never zero

    exp_terms = exp_terms / sum_exp

    z_bar = np.sum(z_clean_reshaped * exp_terms, axis=1)
    z_bar = to_tens(z_bar, device=device)

    z_batch = z_batch.requires_grad_(True)
    d = dnet(z_batch)
    grad_d, norm_d = grad_d_z(z_batch, d)

    # Add small epsilon to prevent division by zero in gradient normalization
    norm_d_safe = torch.clamp(norm_d, min=1e-12)
    normalized_grad = grad_d / norm_d_safe.unsqueeze(1)

    z_shift = z_batch - torch.abs(d) * normalized_grad

    d_val = torch.mean(torch.abs(d)).item()

    return z_bar, z_shift, d_val

def dae_reconstruction(images, model, device):
    with torch.no_grad():
        output = model(images)
        return output

def ldm_reconstruction(images, dae_model, diffusion_model, device, **kwargs):
    diffusion_steps = kwargs['diffusion_steps']
    latents = dae_model.encoder(images)
    degraded_latent= latents[0] if isinstance(latents, tuple) else latents
    skip_features= latents[1] if isinstance(latents, tuple) else None
    #degraded_latent = dae_model.encoder(images)
    denoised_reconstruction, _ = p_sample_loop(
            diffusion_model,
            dae_model,
            latent_start=degraded_latent,
            diffusion_steps_used=diffusion_steps,
            skip_features=skip_features
        )
    return denoised_reconstruction

# XXXXXXXXXXXXXXXXXXXXXXXXXXXX
def reconstruction_with_iterations(images, model, dnet, device, method, train_loader, save_iterations=False, **kwargs):
    """
    Modified reconstruction function that can save intermediate iterations
    """
    eval_cfg = kwargs.get('eval_cfg', None)
    n_iters = eval_cfg['n_iters']
    n_iters = 5* n_iters  # Scale up iterations for more detailed process
    fuse_lambda = eval_cfg['fuse_lambda'] / 5
    delta_fuse = eval_cfg['delta_fuse'] / 5

    sigma = eval_cfg['sigma']
    d_thresh = eval_cfg.get('d_thresh', None)

    z_batch, z_bar, z_shift, d_val = calc_z_components(images, model, dnet, device, method, train_loader, **kwargs)

    if method == 'L2':
        with torch.no_grad():
            output = model.decoder(z_batch)
            if save_iterations:
                return output, [output]  # Return final and list with just final
            return output

    train_images, _ = next(iter(train_loader))
    train_images = train_images.to(device)

    latents_clean = model.encoder(train_images)
    z_clean = latents_clean[0] if isinstance(latents_clean, tuple) else latents_clean

    latents = model.encoder(images)
    skip_features = latents[1] if isinstance(latents_clean, tuple) else None

    dnet.eval()

    # Store intermediate results if requested
    iteration_results = []
    iteration_results.append(images.clone())
    if save_iterations:
        # Save initial state (iteration 0)
        with torch.no_grad():
            if skip_features is not None:
                initial_output = model.decoder(z_batch, skip_features)
            else:
                initial_output = model.decoder(z_batch)
            initial_output = torch.clamp(initial_output, min=0.0, max=1.0)
            iteration_results.append(initial_output.clone())


    beta_vec = np.arange(1, n_iters + 1, 1) / n_iters

    for ii in range(n_iters):
        if d_thresh is not None and d_val > d_thresh:
            delta = z_batch - z_shift
        else:
            delta = fuse_lambda * (z_batch - z_shift) + (1 - fuse_lambda) * (z_batch - z_bar)

        z_batch = z_batch - beta_vec[ii] * delta
        z_bar, z_shift, d_val = update_z(z_batch, dnet, z_clean, sigma, device)
        fuse_lambda = min(1, fuse_lambda + delta_fuse)

        if skip_features is not None:
            output_tmp = model.decoder(z_batch, skip_features)
        else:
            output_tmp = model.decoder(z_batch)

        output_tmp = torch.clamp(output_tmp, min=0.0, max=1.0)

        # Save iteration result if requested
        if save_iterations:
            iteration_results.append(output_tmp.clone())

        latents = model.encoder(output_tmp)
        skip_features = latents[1] if isinstance(latents, tuple) else None

    with torch.no_grad():
        if skip_features is not None:
            output = model.decoder(z_batch, skip_features)
        else:
            output = model.decoder(z_batch)

    output = torch.clamp(output, min=0.0, max=1.0)

    if save_iterations:
        return output, iteration_results
    return output
#
#
# def visualize_iterative_process(digit_list, model_list, dnet_list, test_loader, device,
#                                 train_loader_list=None, **kwargs):
#     """
#     Visualize the iterative reconstruction process for Gaussian noise
#     """
#     eval_cfg = kwargs.get('eval_cfg', None)
#     num_images = min(eval_cfg['num_images'], 5)  # Limit for visualization
#     n_iters = eval_cfg['n_iters']
#
#     file_path = kwargs['file_path']
#     edge_color = kwargs.get('edge_color', 'blue')
#
#     # Get test images for shape reference
#     test_images, _ = next(iter(test_loader))
#     test_images = test_images.to(device)
#
#     # Generate Gaussian noise
#     gaussian_noise = generate_gaussian_noise_images(test_images[:num_images], "uniform", seed=42)
#
#     # For each digit model, show the iterative process
#     for digit in digit_list:
#         model = model_list[digit]
#         dnet = dnet_list[digit]
#
#         print(f'Visualizing iterative process for digit {digit}')
#
#         # Get reconstruction with all iterations saved
#         final_output, all_iterations = reconstruction_with_iterations(
#             gaussian_noise, model, dnet, device, eval_cfg['method'],
#             train_loader_list[digit], save_iterations=True, **kwargs
#         )
#
#         # Create visualization matrix
#         # Rows: iterations (0 to n_iters), Columns: different images
#         total_iterations = len(all_iterations)
#         vis_matrix = np.zeros((total_iterations * num_images, 28, 28))  # Assuming MNIST 28x28
#
#         for iter_idx, iteration_output in enumerate(all_iterations):
#             iter_np = to_np(iteration_output).squeeze()
#             start_row = iter_idx * num_images
#             end_row = start_row + num_images
#             vis_matrix[start_row:end_row, :, :] = iter_np[:num_images, :, :]
#
#         vis_matrix = np.clip(vis_matrix, 0, 1)
#
#         # Create captions for iterations
#         captions = []
#         for iter_idx in range(total_iterations):
#             if iter_idx == 0:
#                 iter_name = "Initial"
#             elif iter_idx == total_iterations - 1:
#                 iter_name = "Final"
#             else:
#                 iter_name = f"Iter {iter_idx}"
#
#             for img_idx in range(num_images):
#                 captions.append(f"{iter_name}")
#
#         # Save iteration visualization
#         iteration_file = f'{file_path}_digit_{digit}_iterations.png'
#         plot_multi_images(vis_matrix, captions=captions, figs_in_line=num_images,
#                           path=iteration_file, edge_color=edge_color)
#
#         print(f'Saved iteration visualization: {iteration_file}')
#
#
# def create_evolution_grid(digit_list, model_list, dnet_list, test_loader, device,
#                           train_loader_list=None, **kwargs):
#     """
#     Create a comprehensive grid showing evolution across iterations and different noise samples
#     """
#     eval_cfg = kwargs.get('eval_cfg', None)
#     n_iters = eval_cfg['n_iters']
#     file_path = kwargs['file_path']
#     edge_color = kwargs.get('edge_color', 'blue')
#
#     # Get test images for shape reference
#     test_images, _ = next(iter(test_loader))
#     test_images = test_images.to(device)
#
#     # Generate multiple Gaussian noise samples
#     num_noise_samples = 4
#     noise_samples = []
#     for i in range(num_noise_samples):
#         noise = generate_gaussian_noise_images(test_images[:1], "uniform", seed=42 + i * 100)
#         noise_samples.append(noise)
#
#     for digit in digit_list:
#         model = model_list[digit]
#         dnet = dnet_list[digit]
#
#         print(f'Creating evolution grid for digit {digit}')
#
#         # Collect all iterations for all noise samples
#         all_evolutions = []
#
#         for noise_idx, noise in enumerate(noise_samples):
#             final_output, iterations = reconstruction_with_iterations(
#                 noise, model, dnet, device, eval_cfg['method'],
#                 train_loader_list[digit], save_iterations=True, **kwargs
#             )
#             all_evolutions.append(iterations)
#
#         # Create comprehensive grid
#         # Rows: noise samples, Columns: iterations
#         total_iterations = len(all_evolutions[0])
#         grid_matrix = np.zeros((num_noise_samples * total_iterations, 28, 28))
#
#         captions = []
#         row_idx = 0
#
#         for noise_idx in range(num_noise_samples):
#             for iter_idx in range(total_iterations):
#                 iter_output = all_evolutions[noise_idx][iter_idx]
#                 iter_np = to_np(iter_output).squeeze()
#                 if len(iter_np.shape) == 3:  # Multiple images in batch
#                     grid_matrix[row_idx, :, :] = iter_np[0, :, :]
#                 else:  # Single image
#                     grid_matrix[row_idx, :, :] = iter_np[:, :]
#
#                 if iter_idx == 0:
#                     caption = f"N{noise_idx + 1}-Init"
#                 elif iter_idx == total_iterations - 1:
#                     caption = f"N{noise_idx + 1}-Final"
#                 else:
#                     caption = f"N{noise_idx + 1}-I{iter_idx}"
#                 captions.append(caption)
#                 row_idx += 1
#
#         grid_matrix = np.clip(grid_matrix, 0, 1)
#
#         # Save evolution grid
#         evolution_file = f'{file_path}_digit_{digit}_evolution_grid.png'
#         plot_multi_images(grid_matrix, captions=captions, figs_in_line=total_iterations,
#                           path=evolution_file, edge_color=edge_color)
#
#         print(f'Saved evolution grid: {evolution_file}')
#
#
# def create_single_noise_evolution(digit_list, model_list, dnet_list, test_loader, device,
#                                   train_loader_list=None, **kwargs):
#     """
#     Create a simple horizontal evolution showing one noise sample through all iterations
#     """
#     eval_cfg = kwargs.get('eval_cfg', None)
#     file_path = kwargs['file_path']
#     edge_color = kwargs.get('edge_color', 'blue')
#
#     # Get test images for shape reference
#     test_images, _ = next(iter(test_loader))
#     test_images = test_images.to(device)
#
#     # Generate one Gaussian noise sample
#     gaussian_noise = generate_gaussian_noise_images(test_images[:1], "", seed=42)
#
#     for digit in digit_list:
#         model = model_list[digit]
#         dnet = dnet_list[digit]
#
#         print(f'Creating single evolution for digit {digit}')
#
#         # Get all iterations
#         final_output, all_iterations = reconstruction_with_iterations(
#             gaussian_noise, model, dnet, device, eval_cfg['method'],
#             train_loader_list[digit], save_iterations=True, **kwargs
#         )
#
#         # Create horizontal evolution
#         total_iterations = len(all_iterations)
#         evolution_matrix = np.zeros((total_iterations, 28, 28))
#
#         captions = []
#         for iter_idx, iteration_output in enumerate(all_iterations):
#             iter_np = to_np(iteration_output).squeeze()
#             if len(iter_np.shape) == 3:  # Multiple images in batch
#                 evolution_matrix[iter_idx, :, :] = iter_np[0, :, :]
#             else:  # Single image
#                 evolution_matrix[iter_idx, :, :] = iter_np[:, :]
#
#             if iter_idx == 0:
#                 captions.append("Initial")
#             elif iter_idx == total_iterations - 1:
#                 captions.append("Final")
#             else:
#                 captions.append(f"Iter {iter_idx}")
#
#         evolution_matrix = np.clip(evolution_matrix, 0, 1)
#
#         # Save single evolution
#         single_file = f'{file_path}_digit_{digit}_single_evolution.png'
#         plot_multi_images(evolution_matrix, captions=captions, figs_in_line=total_iterations,
#                           path=single_file, edge_color=edge_color)
#
#         print(f'Saved single evolution: {single_file}')
#
#
# # Modified main evaluation function that includes iterative visualization
# def eval_all_digits_with_iterations(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
#                                     train_loader_list, test_loader_list, prob_params_list, per_digit=True, device='cpu'):
#     """
#     Main evaluation function that includes iterative process visualization
#     """
#     import os
#     for digit in digit_list:
#         print(f'Evaluating model for digit {digit} with iteration visualization')
#
#         if per_digit:
#             from_digits = [digit]
#             file_path = os.path.join(figure_path, f'per_digit_{digit}')
#         else:
#             from_digits = digit_list
#             file_path = os.path.join(figure_path, f'all_digits_{digit}')
#
#         # Regular visualization
#         visualize_results_multi_digits(
#             from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
#             train_loader_list=train_loader_list,
#             file_path=file_path,
#             prob_params=prob_params_list[digit], digit=digit, edge_color='midnightblue', eval_cfg=eval_cfg
#         )
#
#         # Iterative process visualizations
#         visualize_iterative_process(
#             from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
#             train_loader_list=train_loader_list,
#             file_path=file_path,
#             prob_params=prob_params_list[digit], digit=digit, edge_color='red', eval_cfg=eval_cfg
#         )
#
#         create_evolution_grid(
#             from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
#             train_loader_list=train_loader_list,
#             file_path=file_path,
#             prob_params=prob_params_list[digit], digit=digit, edge_color='green', eval_cfg=eval_cfg
#         )
#
#         create_single_noise_evolution(
#             from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
#             train_loader_list=train_loader_list,
#             file_path=file_path,
#             prob_params=prob_params_list[digit], digit=digit, edge_color='purple', eval_cfg=eval_cfg
#         )


# Usage: Replace your eval_all_digits call with:
# eval_all_digits_with_iterations(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
#                                train_loader_list, test_loader_list, prob_params_list, per_digit=per_digit)

# ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
def visualize_iterative_process_from_noise(digit_list, model_list, dnet_list, test_loader, device,
                                           train_loader_list=None, **kwargs):
    """
    Visualize the iterative reconstruction process starting from PURE GAUSSIAN NOISE
    """
    eval_cfg = kwargs.get('eval_cfg', None)
    num_images = min(eval_cfg['num_images'], 5)  # Limit for visualization
    n_iters = eval_cfg['n_iters']

    file_path = kwargs['file_path']
    edge_color = kwargs.get('edge_color', 'blue')

    # Get test images ONLY for shape reference (28x28)
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    # *** THIS IS THE KEY FIX ***
    # Generate PURE GAUSSIAN NOISE - not using test images as input!
    gaussian_noise = torch.randn_like(test_images[:num_images]) * 0.5 + 0.5
    del test_images
    gaussian_noise = torch.clamp(gaussian_noise, 0, 1)

    # For each digit model, show the iterative process
    for digit in digit_list:
        model = model_list[digit]
        dnet = dnet_list[digit]

        print(f'Visualizing iterative process for digit {digit} starting from PURE NOISE')

        # Get reconstruction with all iterations saved - starting from NOISE
        final_output, all_iterations = reconstruction_with_iterations(
            gaussian_noise, model, dnet, device, eval_cfg['method'],
            train_loader_list[digit], save_iterations=True, **kwargs
        )

        # Create visualization matrix
        # Rows: iterations (0 to n_iters), Columns: different noise samples
        total_iterations = len(all_iterations)
        vis_matrix = np.zeros((total_iterations * num_images, 28, 28))

        for iter_idx, iteration_output in enumerate(all_iterations):
            iter_np = to_np(iteration_output).squeeze()
            start_row = iter_idx * num_images
            end_row = start_row + num_images
            vis_matrix[start_row:end_row, :, :] = iter_np[:num_images, :, :]

        vis_matrix = np.clip(vis_matrix, 0, 1)

        # Create captions for iterations
        captions = []
        for iter_idx in range(total_iterations):
            if iter_idx == 0:
                iter_name = "Initial (NOISE)"
            elif iter_idx == total_iterations - 1:
                iter_name = "Final"
            else:
                iter_name = f"Iter {iter_idx}"

            for img_idx in range(num_images):
                captions.append(f"{iter_name}")

        # Save iteration visualization
        iteration_file = f'{file_path}_digit_{digit}_noise_to_digit_evolution.png'
        plot_multi_images(vis_matrix, captions=captions, figs_in_line=num_images,
                          path=iteration_file, edge_color=edge_color)

        print(f'Saved noise-to-digit evolution: {iteration_file}')


def create_single_noise_to_digit_evolution(digit_list, model_list, dnet_list, test_loader, device,
                                           train_loader_list=None, **kwargs):
    """
    Create a simple horizontal evolution showing one NOISE sample evolving to digit
    """
    eval_cfg = kwargs.get('eval_cfg', None)
    file_path = kwargs['file_path']
    edge_color = kwargs.get('edge_color', 'blue')

    # Get test images for shape reference only
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    # *** GENERATE PURE GAUSSIAN NOISE - NOT using test images! ***
    pure_noise = torch.randn_like(test_images[:1]) * 0.5 + 0.5  # Single noise sample
    pure_noise = torch.clamp(pure_noise, 0, 1)

    for digit in digit_list:
        model = model_list[digit]
        dnet = dnet_list[digit]

        print(f'Creating single noise evolution for digit {digit} - STARTING FROM PURE NOISE')

        # Get all iterations starting from PURE NOISE
        final_output, all_iterations = reconstruction_with_iterations(
            pure_noise, model, dnet, device, eval_cfg['method'],
            train_loader_list[digit], save_iterations=True, **kwargs
        )

        # Create horizontal evolution
        total_iterations = len(all_iterations)
        evolution_matrix = np.zeros((total_iterations, 28, 28))

        captions = []
        for iter_idx, iteration_output in enumerate(all_iterations):
            iter_np = to_np(iteration_output).squeeze()
            if len(iter_np.shape) == 3:  # Multiple images in batch
                evolution_matrix[iter_idx, :, :] = iter_np[0, :, :]
            else:  # Single image
                evolution_matrix[iter_idx, :, :] = iter_np[:, :]

            if iter_idx == 0:
                captions.append("NOISE")
            elif iter_idx == total_iterations - 1:
                captions.append(f"DIGIT {digit}")
            else:
                captions.append(f"Iter {iter_idx}")

        evolution_matrix = np.clip(evolution_matrix, 0, 1)

        # Save single evolution
        single_file = f'{file_path}_digit_{digit}_noise_to_digit_single.png'
        plot_multi_images(evolution_matrix, captions=captions, figs_in_line=total_iterations,
                          path=single_file, edge_color=edge_color)

        print(f'Saved single noise-to-digit evolution: {single_file}')


def generate_pure_gaussian_noise(shape, device, noise_type="standard", seed=42):
    """
    Generate pure Gaussian noise with specified characteristics
    """
    if seed is not None:
        torch.manual_seed(seed)

    if noise_type == "standard":
        # Standard Gaussian noise scaled to [0,1]
        noise = torch.randn(shape, device=device) * 0.5 + 0.5
    elif noise_type == "uniform":
        # Uniform random noise
        noise = torch.rand(shape, device=device)
    elif noise_type == "high_variance":
        # Higher variance Gaussian
        noise = torch.randn(shape, device=device) * 0.7 + 0.5
    else:
        # Default: standard
        noise = torch.randn(shape, device=device) * 0.5 + 0.5

    return torch.clamp(noise, 0, 1)


# FIXED VERSION of your main evaluation function
def eval_all_digits_with_pure_noise_evolution(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
                                              train_loader_list, test_loader_list, prob_params_list,
                                              per_digit=True, device='cpu'):
    """
    Evaluation function that shows evolution from PURE NOISE to clean digits
    """
    import os
    for digit in digit_list:
        print(f'Evaluating model for digit {digit} - PURE NOISE TO DIGIT EVOLUTION')

        if per_digit:
            from_digits = [digit]
            file_path = os.path.join(figure_path, f'per_digit_{digit}')
        else:
            from_digits = digit_list
            file_path = os.path.join(figure_path, f'all_digits_{digit}')

        # Pure noise to digit evolution visualizations
        visualize_iterative_process_from_noise(
            from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
            train_loader_list=train_loader_list,
            file_path=file_path,
            prob_params=prob_params_list[digit], digit=digit, edge_color='red', eval_cfg=eval_cfg
        )

        create_single_noise_to_digit_evolution(
            from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
            train_loader_list=train_loader_list,
            file_path=file_path,
            prob_params=prob_params_list[digit], digit=digit, edge_color='purple', eval_cfg=eval_cfg
        )


# Replace your existing call with this:
# eval_all_digits_with_pure_noise_evolution(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
#                                          train_loader_list, test_loader_list, prob_params_list,
#                                          per_digit=per_digit, device=device)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def reconstruction(images, model, dnet, device, method, train_loader, **kwargs):

    eval_cfg = kwargs.get('eval_cfg', None)
    n_iters=eval_cfg['n_iters']
    fuse_lambda = eval_cfg['fuse_lambda']
   
    delta_fuse=eval_cfg['delta_fuse']
    sigma=eval_cfg['sigma']
    d_thresh = eval_cfg.get('d_thresh', None)
    z_batch, z_bar, z_shift, d_val = calc_z_components(images, model, dnet, device, method, train_loader, **kwargs)
    
    if method == 'L2':
        with torch.no_grad():
            output = model.decoder(z_batch)
            return output
        
    
    train_images, _ = next(iter(train_loader))
    train_images = train_images.to(device)

    latents_clean = model.encoder(train_images)
    z_clean = latents_clean[0] if isinstance(latents_clean, tuple) else latents_clean

    latents = model.encoder(images)
    skip_features = latents[1] if isinstance(latents, tuple) else None
   
   
    dnet.eval()
    
    beta_vec = np.arange(1,n_iters+1,1)/n_iters
    for ii in range(n_iters):

        if d_thresh is not None and d_val > d_thresh:
            delta = z_batch - z_shift
        else:
            delta = fuse_lambda * (z_batch - z_shift) + (1 - fuse_lambda) * (z_batch - z_bar)
        z_batch = z_batch - beta_vec[ii] * delta
        z_bar, z_shift, d_val = update_z(z_batch, dnet, z_clean, sigma, device)
        fuse_lambda = min(1, fuse_lambda + delta_fuse)
        # output_tmp=model.decoder(z_batch,skip_features)

        if skip_features is not None:
            print(f"skip_features={skip_features.shape}")
            output_tmp = model.decoder(z_batch,skip_features)
        else:
            output_tmp = model.decoder(z_batch)

        output_tmp = torch.clamp(output_tmp, min=0.0, max=1.0)
        latents = model.encoder(output_tmp)
        skip_features = latents[1] if isinstance(latents, tuple) else None
       

    with torch.no_grad():
        if skip_features is not None:
            print(f"skip_features={skip_features.shape}")
            output=model.decoder(z_batch,skip_features)
        else:    
            output=model.decoder(z_batch)

    output = torch.clamp(output, min=0.0, max=1.0)
    #output = (output - output.min()) / (output.max() - output.min())
    return output
    
def calculate_digit_metrics(test_loader,train_loader,model_AE,model_dnet,
                                      model_dae,model_diffusion,device,deformation_list, **kwargs):
    
    noise_factor=1.2
    kernel=7
    alpha=34.0
    sigma=1.1
    min_angle=120
    max_angle=120
    sr_scale=0.22

    # CORRECT - generates pure Gaussian noise
    gaussian_noise = torch.randn_like(test_images[:num_images]) * 0.5 + 0.5
    gaussian_noise = torch.clamp(gaussian_noise, 0, 1)

    results={}    

    for deformation in deformation_list:


        if deformation=='noise':
        
            noisy_images = add_noise(test_images,noise_factor=noise_factor)
        elif deformation=='blur':
        
            noisy_images = add_blur(test_images,kernel=kernel)
        elif deformation=='rotate': 
        
            noisy_images = add_rotate(test_images,min_angle=min_angle,max_angle=max_angle)
        elif deformation=='elastic':
        
            noisy_images = add_elastic_transform(test_images,alpha=alpha,sigma=sigma)
        elif deformation=='SR':    
            noisy_images = add_down_sample(test_images,scale_factor=sr_scale)
        
        reco_images_ours = reconstruction(noisy_images, model_AE, model_dnet, device, 'dist', train_loader, **kwargs)
        mse_ours,psnr_ours,ssim_ours = calc_metrics(test_images,reco_images_ours)
        print(f'Ours: MSE: {mse_ours:.2f} PSNR: {psnr_ours:.2f}, SSIM: {ssim_ours:.2f}')

        reco_images_dae = dae_reconstruction(noisy_images, model_dae, device)
        mse_dae,psnr_dae,ssim_dae = calc_metrics(test_images,reco_images_dae)
        #print(f'DAE: MSE: {mse_dae:.2f} PSNR: {psnr_dae:.2f}, SSIM: {ssim_dae:.2f}')

        reco_images_diffusion = ldm_reconstruction(noisy_images, model_dae, model_diffusion, device, **kwargs)
        mse_diffusion,psnr_diffusion,ssim_diffusion = calc_metrics(test_images,reco_images_diffusion)
        #print(f'Diffusion: MSE: {mse_diffusion:.2f} PSNR: {psnr_diffusion:.2f}, SSIM: {ssim_diffusion:.2f}')
        # results[deformation] = {'ours_mse':(mse_dae,psnr_dae,ssim_dae),
        #                         'dae':(mse_dae,psnr_dae,ssim_dae),
        #                         'diffusion':(mse_diffusion,psnr_diffusion,ssim_diffusion)}
        
        results[deformation]={'ours_mse':mse_ours,
                                'ours_psnr':psnr_ours,
                                'ours_ssim':ssim_ours,
                                'dae_mse':mse_dae,
                                'dae_psnr':psnr_dae,
                                'dae_ssim':ssim_dae,
                                'diffusion_mse':mse_diffusion,
                                'diffusion_psnr':psnr_diffusion,
                                'diffusion_ssim':ssim_diffusion}

    table = PrettyTable()
    table.field_names = ["Deformation", "dae_ssim", "diffusion_ssim", "ours_ssim"]


    
    for deformation, error_dict in results.items():
        table.add_row([
        deformation, 
        f"{error_dict['dae_ssim']:.2f}", 
        f"{error_dict['diffusion_ssim']:.2f}", 
        f"{error_dict['ours_ssim']:.2f}"
        ])
    print(table)    


def calc_metrics_tensor(test_images,reco_images):
    pass


def calc_metrics(test_images,reco_images):
    if isinstance(test_images, torch.Tensor):
        test_images_np = to_np(test_images).squeeze()
    else:
        test_images_np = test_images
    if isinstance(reco_images, torch.Tensor):    
        reco_images_np = to_np(reco_images).squeeze()
    else:
        reco_images_np = reco_images
    
    mse =  [np.linalg.norm(test_images_np[i]-reco_images_np[i]) for i in range(test_images_np.shape[0])]
    mse_value = np.mean(mse)
    psnr_values = [psnr(test_images_np[i], reco_images_np[i], data_range=1.0) for i in range(test_images_np.shape[0])]
    psnr_value = np.mean(psnr_values)
    if test_images.shape[1] == 1: #grayscale
        ssim_values = [ssim(test_images_np[i], reco_images_np[i], data_range=1.0) for i in range(test_images_np.shape[0])]
    else:
        ssim_values = [
        ssim(test_images_np[i], reco_images_np[i],  multichannel=True, data_range=1.0, win_size=3,gaussian_weights=True, sigma=1.5,
             use_sample_covariance=False,
            channel_axis=-1) for i in range(test_images_np.shape[0])]
        ssim_values = np.array(ssim_values)

    ssim_value = np.mean(ssim_values)
    return mse_value,psnr_value,ssim_value


def calc_z_error(model, dnet, test_images, noisy_images, reco_images, prob_params):
    z_test = model.encoder(test_images)
    z_reco = model.encoder(reco_images)
    z_noisy = model.encoder(noisy_images)
    z_test_np = to_np(z_test).squeeze()
    z_reco_np = to_np(z_reco).squeeze()
    z_noisy_np = to_np(z_noisy).squeeze()

    reco_images_np = to_np(reco_images).squeeze()
    test_images_np = to_np(test_images).squeeze()
    noisy_images_np = to_np(noisy_images).squeeze()
    mahalanobis = calc_mahalanobis(z_reco_np,  prob_params['z_mean'], prob_params['inv_cov'])
    err4 = mahalanobis
    #prob = calc_probability(z_reco_np, prob_params['z_mean'], prob_params['z_cov'],prob_params['inv_cov'])
    err3 = np.abs(to_np(dnet(z_reco)).squeeze())
    # np.abs(to_np(dist_from_manif).squeeze())
    # z_error = np.linalg.norm(z_test_np - z_reco_np, axis=1)
    err1 = np.linalg.norm(noisy_images_np-reco_images_np, axis=(1,2))
    err2 = np.linalg.norm(z_noisy_np - z_reco_np, axis=1)
    

    return err4


def generate_gaussian_noise_images(test_images, noise_type="", seed=None):
    """
    Generate Gaussian noise images with same shape as test_images
    """
    if seed is not None:
        torch.manual_seed(seed)

    batch_size, channels, height, width = test_images.shape
    device = test_images.device

    if noise_type == "uniform":
        # Standard Gaussian noise scaled to [0,1]
        noise_images = torch.randn_like(test_images) * 0.5 + 0.5
    elif noise_type == "structured":
        # Lower variance Gaussian noise
        noise_images = torch.randn_like(test_images) * 0.3 + 0.5
    elif noise_type == "high_variance":
        # Higher variance Gaussian noise
        noise_images = torch.randn_like(test_images) * 0.7 + 0.5
    elif noise_type == "pure_random":
        # Pure random uniform noise
        noise_images = torch.rand_like(test_images)
    else:
        # Default: standard Gaussian
        noise_images = torch.randn_like(test_images) * 0.5 + 0.5

    # Clip to valid image range [0, 1]
    noise_images = torch.clamp(noise_images, 0, 1)
    return noise_images


def visualize_results_multi_digits(digit_list, model_list, dnet_list, test_loader, device,
                                   train_loader_list=None, **kwargs):
    """
    Modified visualization function that uses Gaussian noise as input instead of corrupted real images
    """
    eval_cfg = kwargs.get('eval_cfg', None)
    num_images = eval_cfg['num_images']
    method = eval_cfg['method']
    offset = eval_cfg['offset']

    file_path = kwargs['file_path']
    prob_params = kwargs['prob_params']
    cur_digit = kwargs['digit']
    edge_color = kwargs['edge_color']
    n_models = len(digit_list)

    reconstructed_image_np = list(range(n_models))
    denoised_image_np = list(range(n_models))
    refined_image_np = list(range(n_models))
    generated_image_np = list(range(n_models))

    # Get test images for shape reference only
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    test_image_np = to_np(test_images).squeeze()

    n_images = test_images.shape[0]
    final_reconstructed_images = np.zeros((n_images, test_image_np.shape[1], test_image_np.shape[2]))
    final_denoised_images = np.zeros((n_images, test_image_np.shape[1], test_image_np.shape[2]))
    final_refined_images = np.zeros((n_images, test_image_np.shape[1], test_image_np.shape[2]))
    final_generated_images = np.zeros((n_images, test_image_np.shape[1], test_image_np.shape[2]))

    err_noise1 = np.zeros((n_models, n_images))
    err_noise2 = np.zeros((n_models, n_images))
    err_noise3 = np.zeros((n_models, n_images))
    err_noise4 = np.zeros((n_models, n_images))

    # Generate 4 different types of Gaussian noise
    gaussian_noise_1 = generate_gaussian_noise_images(test_images, "uniform", seed=42)
    gaussian_noise_2 = generate_gaussian_noise_images(test_images, "structured", seed=123)
    gaussian_noise_3 = generate_gaussian_noise_images(test_images, "high_variance", seed=456)
    gaussian_noise_4 = generate_gaussian_noise_images(test_images, "pure_random", seed=789)

    # Convert to numpy for visualization
    noise_1_np = to_np(gaussian_noise_1).squeeze()
    noise_2_np = to_np(gaussian_noise_2).squeeze()
    noise_3_np = to_np(gaussian_noise_3).squeeze()
    noise_4_np = to_np(gaussian_noise_4).squeeze()

    # Apply reconstruction to each type of Gaussian noise using each model
    for ii, digit in enumerate(digit_list):
        model = model_list[digit]
        dnet = dnet_list[digit]
        model.eval()
        dnet.eval()

        # Reconstruct from Gaussian noise type 1
        reconstructed_images = reconstruction(gaussian_noise_1, model, dnet, device, method, train_loader_list[digit],
                                              **kwargs)
        reconstructed_image_np[ii] = to_np(reconstructed_images).squeeze()
        err = calc_z_error(model, dnet, test_images, gaussian_noise_1, reconstructed_images, prob_params)
        err_noise1[ii, :] = err

        # Reconstruct from Gaussian noise type 2
        denoised_images = reconstruction(gaussian_noise_2, model, dnet, device, method, train_loader_list[digit],
                                         **kwargs)
        denoised_image_np[ii] = to_np(denoised_images).squeeze()
        err = calc_z_error(model, dnet, test_images, gaussian_noise_2, denoised_images, prob_params)
        err_noise2[ii, :] = err

        # Reconstruct from Gaussian noise type 3
        refined_images = reconstruction(gaussian_noise_3, model, dnet, device, method, train_loader_list[digit],
                                        **kwargs)
        refined_image_np[ii] = to_np(refined_images).squeeze()
        err = calc_z_error(model, dnet, test_images, gaussian_noise_3, refined_images, prob_params)
        err_noise3[ii, :] = err

        # Reconstruct from Gaussian noise type 4
        generated_images = reconstruction(gaussian_noise_4, model, dnet, device, method, train_loader_list[digit],
                                          **kwargs)
        generated_image_np[ii] = to_np(generated_images).squeeze()
        err = calc_z_error(model, dnet, test_images, gaussian_noise_4, generated_images, prob_params)
        err_noise4[ii, :] = err

    # Select best reconstruction for each noise type based on lowest error
    noise1_indices = np.argmin(err_noise1, axis=0)
    noise2_indices = np.argmin(err_noise2, axis=0)
    noise3_indices = np.argmin(err_noise3, axis=0)
    noise4_indices = np.argmin(err_noise4, axis=0)

    if len(digit_list) > 1:
        noise1_perc = len(np.where(noise1_indices == cur_digit)[0]) / n_images
        noise2_perc = len(np.where(noise2_indices == cur_digit)[0]) / n_images
        noise3_perc = len(np.where(noise3_indices == cur_digit)[0]) / n_images
        noise4_perc = len(np.where(noise4_indices == cur_digit)[0]) / n_images
        print(
            f'digit:{cur_digit}, Noise1: {noise1_perc:.2f}, Noise2: {noise2_perc:.2f}, Noise3: {noise3_perc:.2f}, Noise4: {noise4_perc:.2f}')

    # Compile final results using best model for each image
    for i in range(n_images):
        model_idx = noise1_indices[i]
        final_reconstructed_images[i, :, :] = reconstructed_image_np[model_idx][i, :, :]

        model_idx = noise2_indices[i]
        final_denoised_images[i, :, :] = denoised_image_np[model_idx][i, :, :]

        model_idx = noise3_indices[i]
        final_refined_images[i, :, :] = refined_image_np[model_idx][i, :, :]

        model_idx = noise4_indices[i]
        final_generated_images[i, :, :] = generated_image_np[model_idx][i, :, :]

    # Calculate metrics
    # mse_vector_1, psnr_vector_1, ssim_vector_1 = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
    # mse_vector_2, psnr_vector_2, ssim_vector_2 = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
    # mse_vector_3, psnr_vector_3, ssim_vector_3 = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
    # mse_vector_4, psnr_vector_4, ssim_vector_4 = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))

    # for i in range(n_images):
    #     mse, psnr, ssim = calc_metrics(test_images.squeeze(), torch.tensor(final_reconstructed_images))
    #     mse_vector_1[i], psnr_vector_1[i], ssim_vector_1[i] = mse, psnr, ssim
    #
    #     mse, psnr, ssim = calc_metrics(test_images.squeeze(), torch.tensor(final_denoised_images))
    #     mse_vector_2[i], psnr_vector_2[i], ssim_vector_2[i] = mse, psnr, ssim
    #
    #     mse, psnr, ssim = calc_metrics(test_images.squeeze(), torch.tensor(final_refined_images))
    #     mse_vector_3[i], psnr_vector_3[i], ssim_vector_3[i] = mse, psnr, ssim
    #
    #     mse, psnr, ssim = calc_metrics(test_images.squeeze(), torch.tensor(final_generated_images))
    #     mse_vector_4[i], psnr_vector_4[i], ssim_vector_4[i] = mse, psnr, ssim
    #
    # # Print results
    # print(
    #     f"Noise Type 1 - MSE: {np.mean(mse_vector_1):.3f}, PSNR: {np.mean(psnr_vector_1):.3f}, SSIM: {np.mean(ssim_vector_1):.3f}")
    # print(
    #     f"Noise Type 2 - MSE: {np.mean(mse_vector_2):.3f}, PSNR: {np.mean(psnr_vector_2):.3f}, SSIM: {np.mean(ssim_vector_2):.3f}")
    # print(
    #     f"Noise Type 3 - MSE: {np.mean(mse_vector_3):.3f}, PSNR: {np.mean(psnr_vector_3):.3f}, SSIM: {np.mean(ssim_vector_3):.3f}")
    # print(
    #     f"Noise Type 4 - MSE: {np.mean(mse_vector_4):.3f}, PSNR: {np.mean(psnr_vector_4):.3f}, SSIM: {np.mean(ssim_vector_4):.3f}")

    # Save visualizations
    # Original test images (for reference)
    res_orig_matrix = test_image_np[offset:offset + num_images, :, :]
    res_orig_matrix = np.clip(res_orig_matrix, 0, 1)
    plot_multi_images(res_orig_matrix, figs_in_line=num_images, path=f'{file_path}_original.png', edge_color=edge_color)

    # Gaussian noise type 1: input and reconstruction
    res_noise1_matrix = np.zeros((2 * num_images, test_image_np.shape[1], test_image_np.shape[2]))
    res_noise1_matrix[0:num_images, :, :] = noise_1_np[offset:offset + num_images, :, :]
    res_noise1_matrix[num_images:2 * num_images, :, :] = final_reconstructed_images[offset:offset + num_images, :, :]
    res_noise1_matrix = np.clip(res_noise1_matrix, 0, 1)
    plot_multi_images(res_noise1_matrix, figs_in_line=num_images, path=f'{file_path}_gaussian_noise1.png',
                      edge_color=edge_color)

    # Gaussian noise type 2: input and reconstruction
    res_noise2_matrix = np.zeros((2 * num_images, test_image_np.shape[1], test_image_np.shape[2]))
    res_noise2_matrix[0:num_images, :, :] = noise_2_np[offset:offset + num_images, :, :]
    res_noise2_matrix[num_images:2 * num_images, :, :] = final_denoised_images[offset:offset + num_images, :, :]
    res_noise2_matrix = np.clip(res_noise2_matrix, 0, 1)
    plot_multi_images(res_noise2_matrix, figs_in_line=num_images, path=f'{file_path}_gaussian_noise2.png',
                      edge_color=edge_color)

    # Gaussian noise type 3: input and reconstruction
    res_noise3_matrix = np.zeros((2 * num_images, test_image_np.shape[1], test_image_np.shape[2]))
    res_noise3_matrix[0:num_images, :, :] = noise_3_np[offset:offset + num_images, :, :]
    res_noise3_matrix[num_images:2 * num_images, :, :] = final_refined_images[offset:offset + num_images, :, :]
    res_noise3_matrix = np.clip(res_noise3_matrix, 0, 1)
    plot_multi_images(res_noise3_matrix, figs_in_line=num_images, path=f'{file_path}_gaussian_noise3.png',
                      edge_color=edge_color)

    # Gaussian noise type 4: input and reconstruction
    res_noise4_matrix = np.zeros((2 * num_images, test_image_np.shape[1], test_image_np.shape[2]))
    res_noise4_matrix[0:num_images, :, :] = noise_4_np[offset:offset + num_images, :, :]
    res_noise4_matrix[num_images:2 * num_images, :, :] = final_generated_images[offset:offset + num_images, :, :]
    res_noise4_matrix = np.clip(res_noise4_matrix, 0, 1)
    plot_multi_images(res_noise4_matrix, figs_in_line=num_images, path=f'{file_path}_gaussian_noise4.png',
                      edge_color=edge_color)

    # Combined visualization showing all 4 noise types and their reconstructions
    combined_matrix = np.zeros((8 * num_images, test_image_np.shape[1], test_image_np.shape[2]))
    combined_matrix[0:num_images, :, :] = noise_1_np[offset:offset + num_images, :, :]
    combined_matrix[num_images:2 * num_images, :, :] = final_reconstructed_images[offset:offset + num_images, :, :]
    combined_matrix[2 * num_images:3 * num_images, :, :] = noise_2_np[offset:offset + num_images, :, :]
    combined_matrix[3 * num_images:4 * num_images, :, :] = final_denoised_images[offset:offset + num_images, :, :]
    combined_matrix[4 * num_images:5 * num_images, :, :] = noise_3_np[offset:offset + num_images, :, :]
    combined_matrix[5 * num_images:6 * num_images, :, :] = final_refined_images[offset:offset + num_images, :, :]
    combined_matrix[6 * num_images:7 * num_images, :, :] = noise_4_np[offset:offset + num_images, :, :]
    combined_matrix[7 * num_images:8 * num_images, :, :] = final_generated_images[offset:offset + num_images, :, :]
    combined_matrix = np.clip(combined_matrix, 0, 1)
    plot_multi_images(combined_matrix, figs_in_line=num_images, path=f'{file_path}_all_gaussian_results.png',
                      edge_color=edge_color)


# Additional helper function for generating pure latent space noise (optional)
def generate_latent_gaussian_samples(model, num_samples, latent_dim, device, seed=42):
    """
    Generate samples directly in latent space and decode them
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        # Generate random latent vectors
        z_samples = torch.randn(num_samples, latent_dim).to(device)

        # Decode to images
        generated_images = model.decoder(z_samples)
        generated_images = torch.clamp(generated_images, 0, 1)

        return generated_images, z_samples


# Function to compare Gaussian noise generation vs latent space generation
def compare_noise_methods(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
                          train_loader_list, test_loader_list, prob_params_list):
    """
    Compare different methods of noise-based generation
    """
    for digit in digit_list:
        print(f'Comparing noise methods for digit {digit}')

        model = model_AE_list[digit]
        dnet = model_dnet_list[digit]
        latent_dim = eval_cfg.get('latent_dim', 15)
        num_samples = eval_cfg.get('num_images', 10)

        # Get test images for shape reference
        test_images, _ = next(iter(test_loader_list[digit]))
        test_images = test_images.to(device)

        # Method 1: Image space Gaussian noise + reconstruction
        gaussian_noise = generate_gaussian_noise_images(test_images[:num_samples], "uniform", seed=42)
        image_space_results = reconstruction(gaussian_noise, model, dnet, device, eval_cfg['method'],
                                             train_loader_list[digit], **{'eval_cfg': eval_cfg})

        # Method 2: Direct latent space generation
        latent_space_results, _ = generate_latent_gaussian_samples(model, num_samples, latent_dim, device, seed=42)

        # Visualize comparison
        comparison_images = np.zeros((4 * num_samples, 28, 28))  # Assuming MNIST 28x28
        comparison_images[0:num_samples, :, :] = to_np(gaussian_noise).squeeze()
        comparison_images[num_samples:2 * num_samples, :, :] = to_np(image_space_results).squeeze()
        comparison_images[2 * num_samples:3 * num_samples, :, :] = np.random.rand(num_samples, 28, 28)  # Placeholder
        comparison_images[3 * num_samples:4 * num_samples, :, :] = to_np(latent_space_results).squeeze()

        comparison_images = np.clip(comparison_images, 0, 1)

        file_path = os.path.join(figure_path, f'noise_method_comparison_digit_{digit}.png')
        plot_multi_images(comparison_images, figs_in_line=num_samples, path=file_path, edge_color='red')

        print(f'Saved comparison for digit {digit}')

# def visualize_results_multi_digits(digit_list, model_list, dnet_list, test_loader, device,
#                                    train_loader_list=None, **kwargs):
#
#
#
#
#     eval_cfg = kwargs.get('eval_cfg', None)
#     noise_factor=eval_cfg['noise_factor']#0.4
#     elastic_alpha=eval_cfg['elastic_alpha']  #34.0
#     elastic_sigma=eval_cfg['elastic_sigma']
#     min_angle=eval_cfg['min_angle']#-80#-100
#     max_angle=eval_cfg['max_angle']#-80#-100
#     sr_scale=eval_cfg['sr_scale']#0.35
#     method=eval_cfg['method']
#     num_images = eval_cfg['num_images']
#     method = eval_cfg['method']
#     offset = eval_cfg['offset']
#
#     file_path = kwargs['file_path']
#     prob_params = kwargs['prob_params']
#     cur_digit=kwargs['digit']
#     edge_color=kwargs['edge_color']
#     n_models = len(digit_list)
#     reconstructed_image_np = list(range(n_models))
#     deblurred_image_np = list(range(n_models))
#     unrotated_image_np = list(range(n_models))
#     unelastic_image_np = list(range(n_models))
#
#     test_images, _ = next(iter(test_loader))
#     test_images = test_images.to(device)
#
#
#     test_image_np = to_np(test_images).squeeze()
#
#
#     n_images = test_images.shape[0]
#     final_denoised_images = np.zeros((n_images,test_image_np.shape[1],test_image_np.shape[2]))
#     final_deblurred_images = np.zeros((n_images,test_image_np.shape[1],test_image_np.shape[2]))
#     final_rotated_images = np.zeros((n_images,test_image_np.shape[1],test_image_np.shape[2]))
#     final_unelastic_images = np.zeros((n_images,test_image_np.shape[1],test_image_np.shape[2]))
#     err_noise = np.zeros((n_models,n_images))
#     err_blur = np.zeros((n_models,n_images))
#     err_rotate = np.zeros((n_models,n_images))
#     err_elastic= np.zeros((n_models,n_images))
#
#
#     noisy_images = add_noise(test_images,noise_factor=noise_factor)
#     blurred_images = add_down_sample(test_images,sr_scale)
#     elastic_images = add_elastic_transform(test_images,alpha=elastic_alpha,sigma=elastic_sigma)
#     rotated_images = add_rotate(test_images,min_angle,max_angle)
#
#     noisy_image_np = to_np(noisy_images).squeeze()
#     blurred_image_np = to_np(blurred_images).squeeze()
#     rotated_image_np = to_np(rotated_images).squeeze()
#     elastic_image_np = to_np(elastic_images).squeeze()
#
#     for ii,digit in enumerate(digit_list):
#         print("Processing digit:", digit)
#         model = model_list[digit]
#         dnet = dnet_list[digit]
#         model.eval()
#         dnet.eval()
#
#         reconstructed_images = reconstruction(noisy_images, model, dnet, device, method, train_loader_list[digit], **kwargs)
#         reconstructed_image_np[ii] = to_np(reconstructed_images).squeeze()
#         err= calc_z_error(model, dnet, test_images, noisy_images, reconstructed_images, prob_params)
#         err_noise[ii,:] = err
#
#
#         deblurred_images = reconstruction(blurred_images, model, dnet, device, method, train_loader_list[digit],**kwargs)
#         deblurred_image_np[ii] = to_np(deblurred_images).squeeze()
#         err = calc_z_error(model, dnet, test_images, blurred_images, deblurred_images,prob_params)
#         err_blur[ii,:] = err
#
#
#         unelastic_images = reconstruction(elastic_images, model, dnet, device, method, train_loader_list[digit],**kwargs)
#         unelastic_image_np[ii] = to_np(unelastic_images).squeeze()
#         err = calc_z_error(model, dnet, test_images, elastic_images, unelastic_images,prob_params)
#         err_elastic[ii,:] = err
#
#
#         unrotated_images = reconstruction(rotated_images, model, dnet, device, method,train_loader_list[digit], **kwargs)
#         unrotated_image_np[ii] = to_np(unrotated_images).squeeze()
#         err = calc_z_error(model, dnet, test_images, rotated_images, unrotated_images, prob_params)
#
#         err_rotate[ii,:] = err
#
#
#     noise_indices = np.argmin(err_noise,axis=0)
#     blur_indices = np.argmin(err_blur,axis=0)
#     rotate_indices = np.argmin(err_rotate,axis=0)
#     elastic_indices = np.argmin(err_elastic,axis=0)
#
#     if len(digit_list)> 1:
#         noise_perc = len(np.where(noise_indices==cur_digit)[0])/n_images
#         sr_perc = len(np.where(blur_indices==cur_digit)[0])/n_images
#         rotate_perc = len(np.where(rotate_indices==cur_digit)[0])/n_images
#         elastic_perc = len(np.where(elastic_indices==cur_digit)[0])/n_images
#         print(f'digit:{cur_digit}, Noise: {noise_perc:.2f}, Blur: {sr_perc:.2f}, Rotate: {rotate_perc:.2f}, Elastic: {elastic_perc:.2f}')
#
#
#
#
#     mse_vector_noise, psnr_vector_noise, ssim_vector_noise = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
#     mse_vector_sr, psnr_vector_sr, ssim_vector_sr = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
#     mse_vector_rotate, psnr_vector_rotate, ssim_vector_rotate = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
#     mse_vector_elastic, psnr_vector_elastic, ssim_vector_elastic = np.zeros((n_images)), np.zeros((n_images)), np.zeros((n_images))
#
#     # for i in range(n_images):
#     #     model_idx = noise_indices[i]
#     #     final_denoised_images[i,:,:] = reconstructed_image_np[model_idx][i,:,:]
#     #     model_idx = blur_indices[i]
#     #     final_deblurred_images[i,:,:] = deblurred_image_np[model_idx][i,:,:]
#     #     model_idx = rotate_indices[i]
#     #     final_rotated_images[i,:,:] = unrotated_image_np[model_idx][i,:,:]
#     #     model_idx = elastic_indices[i]
#     #     final_unelastic_images[i,:,:] = unelastic_image_np[model_idx][i,:,:]
#     #     mse,psnr,ssim = calc_metrics(test_images.squeeze(),torch.tensor(final_denoised_images))
#     #     mse_vector_noise[i] ,psnr_vector_noise[i], ssim_vector_noise[i] = mse,psnr,ssim
#     #     mse,psnr,ssim = calc_metrics(test_images.squeeze(),torch.tensor(final_deblurred_images))
#     #     mse_vector_sr[i] ,psnr_vector_sr[i], ssim_vector_sr[i] = mse,psnr,ssim
#     #     mse,psnr,ssim = calc_metrics(test_images.squeeze(),torch.tensor(final_rotated_images))
#     #     mse_vector_rotate[i] ,psnr_vector_rotate[i], ssim_vector_rotate[i] = mse,psnr,ssim
#     #     mse,psnr,ssim = calc_metrics(test_images.squeeze(),torch.tensor(final_unelastic_images))
#     #     mse_vector_elastic[i] ,psnr_vector_elastic[i], ssim_vector_elastic[i] = mse,psnr,ssim
#
#     mse_noise, psnr_noise,ssim_noise = np.mean(mse_vector_noise),np.mean(psnr_vector_noise),np.mean(ssim_vector_noise)
#     mse_blur, psnr_blur,ssim_blur = np.mean(mse_vector_sr),np.mean(psnr_vector_sr),np.mean(ssim_vector_sr)
#     mse_rotate, psnr_rotate,ssim_rotate = np.mean(mse_vector_rotate),np.mean(psnr_vector_rotate),np.mean(ssim_vector_rotate)
#     mse_elastic, psnr_elastic,ssim_elastic = np.mean(mse_vector_elastic),np.mean(psnr_vector_elastic),np.mean(ssim_vector_elastic)
#
#
#
#     # table = PrettyTable()
#     # table.field_names = ["Deformation", "MSE", "PSNR", "SSIM"]
#
#     # table.add_row(['Noise', f"{mse_noise:.2f}", f"{psnr_noise:.2f}", f"{ssim_noise:.2f}"])
#     # table.add_row(['Blur', f"{mse_blur:.2f}", f"{psnr_blur:.2f}", f"{ssim_blur:.2f}"])
#     # table.add_row(['Rotate', f"{mse_rotate:.2f}", f"{psnr_rotate:.2f}", f"{ssim_rotate:.2f}"])
#     # table.add_row(['Elastic', f"{mse_elastic:.2f}", f"{psnr_elastic:.2f}", f"{ssim_elastic:.2f}"])
#
#
#
#     # print(table)
#
#
#
#     res_orig_matrix = np.zeros((num_images,test_image_np.shape[1],test_image_np.shape[2]))
#     res_orig_matrix = test_image_np[offset:offset+num_images,:,:]
#     res_orig_matrix = np.clip(res_orig_matrix,0,1)
#
#     plot_multi_images(res_orig_matrix,figs_in_line=num_images,path=f'{file_path}_original.png',edge_color=edge_color)
#
#     res_elastic_matrix = np.zeros((2*num_images,test_image_np.shape[1],test_image_np.shape[2]))
#     res_elastic_matrix[0:num_images,:,:] = elastic_image_np[offset:offset+num_images,:,:]
#     res_elastic_matrix[num_images:2*num_images,:,:] = final_unelastic_images [offset:offset+num_images,:,:]
#     res_elastic_matrix = np.clip(res_elastic_matrix,0,1)
#     plot_multi_images(res_elastic_matrix,figs_in_line=num_images,path=f'{file_path}_elastic.png',edge_color=edge_color)
#
#
#     res_rotation_matrix = np.zeros((2*num_images,test_image_np.shape[1],test_image_np.shape[2]))
#     res_rotation_matrix[0:num_images,:,:] = rotated_image_np[offset:offset+num_images,:,:]
#     res_rotation_matrix[num_images:2*num_images,:,:] = final_rotated_images [offset:offset+num_images,:,:]
#     res_rotation_matrix = np.clip(res_rotation_matrix,0,1)
#     plot_multi_images(res_rotation_matrix,figs_in_line=num_images,path=f'{file_path}_rotation.png',edge_color=edge_color)
#
#
#     res_deblurred_matrix = np.zeros((2*num_images,test_image_np.shape[1],test_image_np.shape[2]))
#     res_deblurred_matrix[0:num_images,:,:] = blurred_image_np[offset:offset+num_images,:,:]
#     res_deblurred_matrix[num_images:2*num_images,:,:] = final_deblurred_images [offset:offset+num_images,:,:]
#     res_deblurred_matrix = np.clip(res_deblurred_matrix,0,1)
#     plot_multi_images(res_deblurred_matrix,figs_in_line=num_images,path=f'{file_path}_blur.png',edge_color=edge_color)
#
#     res_denoised_matrix = np.zeros((2*num_images,test_image_np.shape[1],test_image_np.shape[2]))
#     res_denoised_matrix[0:num_images,:,:] = noisy_image_np[offset:offset+num_images,:,:]
#     res_denoised_matrix[num_images:2*num_images,:,:] = final_denoised_images [offset:offset+num_images,:,:]
#     res_denoised_matrix = np.clip(res_denoised_matrix,0,1)
#     plot_multi_images(res_denoised_matrix,figs_in_line=num_images,path=f'{file_path}_noise.png',edge_color=edge_color)

    

      
def visualize_results(model, dnet, test_loader, device, method = 'L2', num_images=5, **kwargs):

    
    model.eval()
    dnet.eval()
    #with torch.no_grad():
    if True:
        # Get a batch of test images
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)

        # Add noise
        noisy_images = add_noise(test_images,noise_factor=0.45)
        blurred_images = add_blur(test_images,9)
        rotated_images = add_rotate(test_images,-90,-100)
        # Reconstruct images
       
        reconstructed_images = reconstruction(noisy_images, model, dnet, device, method,**kwargs)
        deblurred_images = reconstruction(blurred_images, model, dnet, device, method,**kwargs)
        unrotated_images = reconstruction(rotated_images, model, dnet, device, method,**kwargs)
        unrotated_images = reconstruction(unrotated_images, model, dnet, device, method,**kwargs)
        #unrotated_images = reconstruction(unrotated_images, model, dnet, device, method,**kwargs)

        # Plot results
        plt.figure(figsize=(16, 8))
        offset=0
        
        for i in range(num_images):
            ind=i+offset
            test_image_np = to_np(test_images[ind]).squeeze()
            noisy_image_np = to_np(noisy_images[ind]).squeeze()
            reconstructed_image_np = to_np(reconstructed_images[ind]).squeeze()
            rotated_images_np = to_np(rotated_images[ind]).squeeze()
            unrotated_images_np = to_np(unrotated_images[ind]).squeeze()
            blurred_images_np = to_np(blurred_images[ind]).squeeze()
            deblurred_images_np = to_np(deblurred_images[ind]).squeeze()

            # Noisy input
            plt.subplot(7, num_images, i + 1)
            plt.imshow(test_image_np, cmap='gray')
            plt.title('Original')
            plt.axis('off')

            # Original image
            plt.subplot(7, num_images, num_images + i + 1)
            plt.imshow(noisy_image_np, cmap='gray')
            err = np.linalg.norm(noisy_image_np-test_image_np)
            plt.title('noisy {:.2f}'.format(err))
            plt.axis('off')

            # Reconstructed image
            plt.subplot(7, num_images, 2 * num_images + i + 1)
            plt.imshow(reconstructed_image_np, cmap='gray')
            err = np.linalg.norm(reconstructed_image_np - test_image_np)
            plt.title('Denoised {:.2f}'.format(err))
            plt.axis('off')

            plt.subplot(7, num_images, 3 * num_images + i + 1)
            plt.imshow(rotated_images_np, cmap='gray')
            err = np.linalg.norm(rotated_images_np - test_image_np)
            plt.title('Rot {:.2f}'.format(err))
            plt.axis('off')

            plt.subplot(7, num_images, 4 * num_images + i + 1)
            plt.imshow(unrotated_images_np, cmap='gray')
            err = np.linalg.norm(unrotated_images_np - test_image_np)
            plt.title(' {:.2f}'.format(err))
            plt.axis('off')

            plt.subplot(7, num_images, 5 * num_images + i + 1)
            plt.imshow(blurred_images_np, cmap='gray')
            err = np.linalg.norm(blurred_images_np - test_image_np)
            plt.title('Blur {:.2f}'.format(err))
            plt.axis('off')

            plt.subplot(7, num_images, 6 * num_images + i + 1)
            plt.imshow(deblurred_images_np, cmap='gray')
            err = np.linalg.norm(deblurred_images_np - test_image_np)
            plt.title('Deblur {:.2f}'.format(err))
            plt.axis('off')

        plt.tight_layout()    

def visualize_images(images_np, noisy_image_np, denoised_image_np, offset_idx = 0, num_images=10, output_path=None):
    """
    Display original, noisy, and denoised images in a grid with 3 rows and num_images columns.
    """
    num_images = min(num_images, images_np.shape[0])  # Ensure we don't exceed batch size
    
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))  # 3 rows: Original, Noisy, Denoised
    
    for i in range(num_images):
        # Original Image
        axes[0, i].imshow(images_np[i+offset_idx],cmap='gray')
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12, fontweight='bold')

        # Noisy Image
        axes[1, i].imshow(noisy_image_np[i+offset_idx],cmap='gray')
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Noisy", fontsize=12, fontweight='bold')

        # Denoised Image
        axes[2, i].imshow(denoised_image_np[i+offset_idx],cmap='gray')
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Denoised", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)