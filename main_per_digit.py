import os
import numpy as np
import torch, torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from architectures import MNISTConvAutoencoder, MLP, DiffusionModel
from losses import loss_dist_al
from mnist_fid_implementation import run_fid_evaluation
from train import train_al, train_dae_diffusion_model
from utils import visualize_results_multi_digits, calc_mean_cov, to_np, \
    eval_all_digits_with_pure_noise_evolution, reconstruction_with_iterations
from mnist_eval import gradual_deformation, compare_performance
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

transform = transforms.Compose([
    transforms.ToTensor(),
])

SR_SEVERE = 0.22
ROT_SEVERE = -150
EL_SEVERE = 1.1
NOISE_SEVERE = 1.3

SR_MILD = 0.3
ROT_MILD = -80
EL_MILD = 1.5
NOISE_MILD = 0.75

SR_INTER = (SR_MILD + SR_SEVERE) / 2
ROT_INTER = (ROT_MILD + ROT_SEVERE) / 2
EL_INTER = (EL_MILD + EL_SEVERE) / 2
NOISE_INTER = (NOISE_MILD + NOISE_SEVERE) / 2


# Add this function to main_per_digit.py:
def evaluate_fid_scores(digit_list, model_AE_list, model_dnet_list, train_loader_list,
                        device, eval_cfg, per_digit=True):
    """Evaluate FID scores for noise-to-image generation"""

    print("\n" + "=" * 50)
    print("EVALUATING FID SCORES")
    print("=" * 50)

    fid_scores, overall_fid = run_fid_evaluation(
        digit_list, model_AE_list, model_dnet_list, train_loader_list,
        device, eval_cfg, per_digit=per_digit
    )

    if per_digit and fid_scores:
        print("\nPer-digit FID scores:")
        for digit, score in fid_scores.items():
            print(f"Digit {digit}: {score:.2f}")

    print(f"\nOverall FID score: {overall_fid:.2f}")

    return fid_scores, overall_fid


def reload_diff_models(digit_list, model_dae_path_list, model_diff_path_list, model_dae_list, model_diffusion_list):
    for digit in digit_list:
        checkpoint = torch.load(model_dae_path_list[digit], weights_only=True)
        model_dae_list[digit].load_state_dict(checkpoint)
        checkpoint = torch.load(model_diff_path_list[digit], weights_only=True)
        model_diffusion_list[digit].load_state_dict(checkpoint)
    return model_dae_list, model_diffusion_list


def reload_models(digit_list, model_path_list, model_AE_list, model_dnet_list, train_loader_list):
    prob_params_list = list(range(10))
    for digit in digit_list:  # Only process digits in digit_list
        checkpoint = torch.load(model_path_list[digit], weights_only=True)
        model_AE_list[digit].load_state_dict(checkpoint['AE_state_dict'])
        model_dnet_list[digit].load_state_dict(checkpoint['dnet_state_dict'])
        # Only calculate prob_params for digits we actually have
        prob_params_list[digit] = calc_mean_cov(train_loader_list[digit], model_AE_list[digit])

    return model_AE_list, model_dnet_list, prob_params_list


def prepare_data(digit_list, cfg):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transform,
                                              download=True)

    train_ds = list(range(10))
    test_ds = list(range(10))
    train_loader_list = list(range(10))
    test_loader_list = list(range(10))
    model_AE_list = list(range(10))
    model_dnet_list = list(range(10))
    model_dae_list = list(range(10))
    model_diffusion_list = list(range(10))

    latent_dim = cfg['latent_dim']
    for digit in digit_list:
        idx = (train_dataset.targets == digit)
        train_ds[digit] = torch.utils.data.Subset(train_dataset, indices=torch.where(idx)[0])
        idx = (test_dataset.targets == digit)
        test_ds[digit] = torch.utils.data.Subset(test_dataset, indices=torch.where(idx)[0])

        train_loader_list[digit] = torch.utils.data.DataLoader(train_ds[digit], batch_size=cfg['batch_size'],
                                                               shuffle=True)
        test_loader_list[digit] = torch.utils.data.DataLoader(test_ds[digit], batch_size=cfg['batch_size'],
                                                              shuffle=False)

        model_AE_list[digit] = MNISTConvAutoencoder(latent_dim).to(device)
        model_dnet_list[digit] = MLP(latent_dim, cfg['mlp_arch']).to(device)
        model_dae_list[digit] = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        model_diffusion_list[digit] = DiffusionModel(latent_dim).to(device)

    return train_loader_list, test_loader_list, model_AE_list, model_dnet_list, model_dae_list, \
        model_diffusion_list, train_ds, test_ds


def train_mnist(digit_list, cfg, model_AE_list, model_dnet_list, train_loader_list, model_path_list, figure_path):
    for digit in digit_list:
        print('training model for digit {}'.format(digit))
        model_AE_list[digit].train()
        model_dnet_list[digit].train()
        all_images = []
        for images, _ in train_loader_list[digit]:
            all_images.append(images)

        # Concatenate all batches along the first dimension (batch dimension)
        all_images = torch.cat(all_images, dim=0)
        all_images = all_images.to(device)

        optimizer = optim.Adam([
            {'params': model_AE_list[digit].parameters(), 'lr': cfg['lr_ae']},
            {'params': model_dnet_list[digit].parameters(), 'lr': cfg['lr_dnet']}
        ])
        train_losses, lam_list, loss_vector = train_al(model_AE_list[digit], train_loader_list[digit], cfg['loss_func'],
                                                       optimizer,
                                                       device, epochs=cfg['epochs'], dnet=model_dnet_list[digit],
                                                       all_images=all_images,
                                                       lam3=cfg['lam3'], lam6=cfg['lam6'], lam7=cfg['lam7'],
                                                       do_update=cfg['do_update'])
        checkpoint = {
            'AE_state_dict': model_AE_list[digit].state_dict(),
            'dnet_state_dict': model_dnet_list[digit].state_dict()}

        print('saving {}'.format(model_path_list[digit]))
        torch.save(checkpoint, model_path_list[digit])
        plt.figure()
        plt.plot(train_losses)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(figure_path, f'train_loss_digit{digit}.png'))
        plt.figure()
        lam_vec = np.array(lam_list)
        plt.plot(lam_vec, label='lam3')
        plt.title(f'lambda, loss3={loss_vector[2]:.3f} loss4={loss_vector[3]:.3f}')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(figure_path, f'train_lam_digit{digit}.png'))
    plt.close('all')


def train_diffusion_mnist(digit_list, train_loader_list, dae_model_list, diffusion_model_list,
                          device, dae_epochs, diff_epochs, lr=1e-3, diffusion_steps=1200,
                          dae_model_path_list=None, diff_model_path_list=None, min_noise=0.4, max_noise=0.4):
    for digit in digit_list:
        print(f"Training DAE for digit {digit}...")
        train_dae_diffusion_model(digit, train_loader_list[digit], dae_model_list[digit], diffusion_model_list[digit],
                                  device,
                                  dae_epochs, diff_epochs, lr=lr, diffusion_steps=diffusion_steps,
                                  dae_model_path=dae_model_path_list[digit],
                                  diff_model_path=diff_model_path_list[digit], min_noise=min_noise,
                                  max_noise=max_noise)
        print(f"Training Diffusion for digit {digit}...")


def eval_all_digits(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list, train_loader_list,
                    test_loader_list, prob_params_list, per_digit=True):
    for digit in digit_list:
        print('evaluating model for digit {}'.format(digit))
        if per_digit:
            from_digits = [digit]
            file_path = os.path.join(figure_path, f'per_digit_{digit}')
        else:
            from_digits = digit_list
            file_path = os.path.join(figure_path, f'all_digits_{digit}')

        print(file_path)
        visualize_results_multi_digits(
            from_digits, model_AE_list, model_dnet_list, test_loader_list[digit], device,
            train_loader_list=train_loader_list,
            file_path=file_path,
            prob_params=prob_params_list[digit], digit=digit, edge_color='midnightblue', eval_cfg=eval_cfg
        )


def compare_alg_results(digit_list, test_ds, model_AE_list, model_dnet_list, model_dae_list, model_diffusion_list,
                        train_loader_list,
                        eval_cfg, figure_path=None, index=10):
    for digit in digit_list:
        test_image = test_ds[digit][index][0]
        for i, deformation in enumerate(['noise', 'SR', 'rotate', 'elastic']):
            gradual_deformation(digit, test_ds[digit], None, deformation, model_AE_list[digit], model_dnet_list[digit],
                                model_dae_list[digit], model_diffusion_list[digit], train_loader_list[digit],
                                eval_cfg, figure_path, test_image=test_image)


def qualitative_results(digit_list, test_loader_list, model_AE_list, model_dnet_list, model_dae_list, model_diff_list,
                        train_loader_list,
                        eval_cfg, type='severe'):
    ours_ssim_sr_list = []
    diff_ssim_sr_list = []
    dae_ssim_sr_list = []
    ours_ssim_el_list = []
    diff_ssim_el_list = []
    dae_ssim_el_list = []
    ours_ssim_rot_list = []
    diff_ssim_rot_list = []
    dae_ssim_rot_list = []
    ours_ssim_noi_list = []
    diff_ssim_noi_list = []
    dae_ssim_noi_list = []

    # severe
    if type == 'severe':
        sr_level = SR_SEVERE
        rot_level = ROT_SEVERE
        el_level = EL_SEVERE
        noi_level = NOISE_SEVERE

    # intermediate
    if type == 'intermediate':
        sr_level = SR_INTER
        rot_level = ROT_INTER
        el_level = EL_INTER
        noi_level = NOISE_INTER

    if type == 'mild':
        sr_level = SR_MILD
        rot_level = ROT_MILD
        el_level = EL_MILD
        noi_level = NOISE_MILD

    for digit in digit_list:
        table = PrettyTable()
        table.field_names = ["Method", "noise", "rotate", "elastic", "SR"]

        ours_ssim_sr, dae_ssim_sr, diff_ssim_sr = compare_performance(digit, test_loader_list[digit], 'SR', sr_level,
                                                                      model_AE_list[digit], model_dnet_list[digit],
                                                                      model_dae_list[digit], model_diff_list[digit],
                                                                      train_loader_list[digit], eval_cfg=eval_cfg)

        ours_ssim_rot, dae_ssim_rot, diff_ssim_rot = compare_performance(digit, test_loader_list[digit], 'rotate',
                                                                         rot_level, model_AE_list[digit],
                                                                         model_dnet_list[digit],
                                                                         model_dae_list[digit], model_diff_list[digit],
                                                                         train_loader_list[digit], eval_cfg=eval_cfg)

        ours_ssim_el, dae_ssim_el, diff_ssim_el = compare_performance(digit, test_loader_list[digit], 'elastic',
                                                                      el_level, model_AE_list[digit],
                                                                      model_dnet_list[digit],
                                                                      model_dae_list[digit], model_diff_list[digit],
                                                                      train_loader_list[digit], eval_cfg=eval_cfg)

        ours_ssim_noi, dae_ssim_noi, diff_ssim_noi = compare_performance(digit, test_loader_list[digit], 'noise',
                                                                         noi_level, model_AE_list[digit],
                                                                         model_dnet_list[digit],
                                                                         model_dae_list[digit], model_diff_list[digit],
                                                                         train_loader_list[digit], eval_cfg=eval_cfg)

        ours_ssim_sr_list.append(ours_ssim_sr)
        diff_ssim_sr_list.append(diff_ssim_sr)
        dae_ssim_sr_list.append(dae_ssim_sr)

        ours_ssim_el_list.append(ours_ssim_el)
        diff_ssim_el_list.append(diff_ssim_el)
        dae_ssim_el_list.append(dae_ssim_el)

        ours_ssim_rot_list.append(ours_ssim_rot)
        diff_ssim_rot_list.append(diff_ssim_rot)
        dae_ssim_rot_list.append(dae_ssim_rot)

        ours_ssim_noi_list.append(ours_ssim_noi)
        diff_ssim_noi_list.append(diff_ssim_noi)
        dae_ssim_noi_list.append(dae_ssim_noi)

        table.add_row(["DAE", f'{np.mean(dae_ssim_noi_list):.2f}', f'{np.mean(dae_ssim_rot_list):.2f}',
                       f'{np.mean(dae_ssim_el_list):.2f}', f'{np.mean(dae_ssim_sr_list):.2f}'])
        table.add_row(["Diffusion", f'{np.mean(diff_ssim_noi_list):.2f}', f'{np.mean(diff_ssim_rot_list):.2f}',
                       f'{np.mean(diff_ssim_el_list):.2f}', f'{np.mean(diff_ssim_sr_list):.2f}'])
        table.add_row(["Ours", f'{np.mean(ours_ssim_noi_list):.2f}', f'{np.mean(ours_ssim_rot_list):.2f}',
                       f'{np.mean(ours_ssim_el_list):.2f}', f'{np.mean(ours_ssim_sr_list):.2f}'])

        print(table)


if __name__ == "__main__":
    digit_list = [2]
    retrain = False
    retrain_diffusion = False
    lam3 = 0
    lam6 = 1
    lam7 = 1
    latent = 15
    # per_digit = False

    train_cfg = {}
    eval_cfg = {}
    train_diff_cfg = {}

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path_list = list(range(10))
    model_diff_path_list = list(range(10))
    model_dae_path_list = list(range(10))

    experiment_name = 'Experiments/MNIST_per_digit_with_eikonal'
    os.makedirs(experiment_name, exist_ok=True)
    model_path = os.path.join(experiment_name, 'Models')
    figure_path = os.path.join(experiment_name, 'Figures')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    if lam3 == 0:
        do_update = False
    else:
        do_update = True

    for digit in digit_list:
        model_path_list[digit] = os.path.join(model_path,
                                              f'dist_set_lam3_{lam3}_lam6_{lam6}_lam7_{lam7}_lat{latent}_digit{digit}.pth')
        model_diff_path_list[digit] = os.path.join(model_path, f'diff_lat{latent}_digit{digit}.pth')
        model_dae_path_list[digit] = os.path.join(model_path, f'dae_lat{latent}_digit{digit}.pth')

    train_cfg['lam3'] = lam3
    train_cfg['lam6'] = lam6
    train_cfg['lam7'] = lam7
    train_cfg['min_noise'] = 0.4
    train_cfg['max_noise'] = 0.4
    train_cfg['do_update'] = do_update
    train_cfg['lr_ae'] = 1e-3
    train_cfg['lr_dnet'] = 1e-5
    train_cfg['latent_dim'] = latent
    train_cfg['batch_size'] = 128
    train_cfg['epochs'] = 120
    train_cfg['mlp_arch'] = [100, 50, 20]
    train_cfg['experiment_name'] = experiment_name
    train_cfg['loss_func'] = loss_dist_al

    train_diff_cfg['min_noise'] = train_cfg['min_noise']
    train_diff_cfg['max_noise'] = train_cfg['max_noise']
    train_diff_cfg['dae_epochs'] = 100
    train_diff_cfg['diff_epochs'] = 100
    train_diff_cfg['lr'] = 1e-3
    train_diff_cfg['latent_dim'] = latent
    train_diff_cfg['diffusion_steps'] = 1200

    eval_cfg['method'] = 'dist'
    eval_cfg['n_iters'] = 8
    eval_cfg['fuse_lambda'] = 0.65
    eval_cfg['delta_fuse'] = (0.9 - eval_cfg['fuse_lambda']) / eval_cfg['n_iters']
    eval_cfg['sigma'] = 1
    eval_cfg['d_thresh'] = None
    eval_cfg['noise_factor'] = NOISE_INTER
    eval_cfg['elastic_alpha'] = 34.0
    eval_cfg['elastic_sigma'] = EL_INTER
    eval_cfg['sr_scale'] = SR_MILD
    eval_cfg['min_angle'] = ROT_MILD
    eval_cfg['max_angle'] = ROT_MILD
    eval_cfg['num_images'] = 10
    eval_cfg['offset'] = 30
    eval_cfg['diffusion_steps'] = 150

    train_loader_list, test_loader_list, model_AE_list, \
        model_dnet_list, model_dae_list, model_diff_list, train_ds, test_ds = prepare_data(digit_list, train_cfg)

    if retrain_diffusion:
        train_diffusion_mnist(digit_list, train_loader_list, model_dae_list, model_diff_list,
                              device, train_diff_cfg['dae_epochs'], train_diff_cfg['diff_epochs'],
                              lr=train_diff_cfg['lr'],
                              diffusion_steps=train_diff_cfg['diffusion_steps'],
                              dae_model_path_list=model_dae_path_list,
                              diff_model_path_list=model_diff_path_list,
                              min_noise=train_diff_cfg['min_noise'], max_noise=train_diff_cfg['max_noise'])

    if retrain:
        train_mnist(digit_list, train_cfg, model_AE_list, model_dnet_list, train_loader_list, model_path_list,
                    model_path)

    model_AE_list, model_dnet_list, prob_params_list = reload_models(
        digit_list, model_path_list, model_AE_list, model_dnet_list, train_loader_list)

    model_dae_list, model_diff_list = reload_diff_models(digit_list, model_dae_path_list, model_diff_path_list,
                                                         model_dae_list, model_diff_list)

    # DEBUG: Check parameters and test reconstruction quality
    print("DEBUG: Checking eval_cfg parameters:")
    print(f"method: {eval_cfg['method']}")
    print(f"n_iters: {eval_cfg['n_iters']}")
    print(f"fuse_lambda: {eval_cfg['fuse_lambda']}")
    print(f"delta_fuse: {eval_cfg['delta_fuse']}")

    # Test your reconstruction on a small sample first
    print("\nDEBUG: Testing reconstruction quality...")
    test_images, _ = next(iter(train_loader_list[2]))
    test_images = test_images.to(device)

    n_images = min(100, len(test_images))
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # Generate pure noise sampled uniformly across the hypercube and projected to sphere
    pure_noise = torch.rand_like(test_images[:n_images]) * 2 - 1  # uniform in [-1,1]
    # Normalize to unit sphere
    norms = pure_noise.view(pure_noise.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
    pure_noise = pure_noise / norms
    # Scale to [0,1] image range
    pure_noise = (pure_noise + 1) * 0.5
    pure_noise = torch.clamp(pure_noise, 0, 1)

    # Test reconstruction
    reconstructed = reconstruction_with_iterations(
        pure_noise, model_AE_list[2], model_dnet_list[2], device,
        'dist', train_loader_list[2],
        save_iterations=False, eval_cfg=eval_cfg
    )
    # Save comparison
    fig, axes = plt.subplots(2, n_images, figsize=(12, 6))
    for i in range(n_images):
        axes[0, i].imshow(pure_noise[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Noise {i}')
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')

    plt.suptitle('FID Debug: Noise to Digit 2 Reconstruction')
    plt.tight_layout()
    plt.savefig('fid_debug_reconstruction.png', dpi=350, bbox_inches='tight')
    plt.close()

    print("Saved debug images to: fid_debug_reconstruction.png")
    print("Check if the reconstructed images look like digit 2!")

    # Now run FID with potentially better parameters for pure noise
    fid_eval_cfg = eval_cfg.copy()
    fid_eval_cfg['n_iters'] = 15  # More iterations for pure noise
    fid_eval_cfg['fuse_lambda'] = 0.3  # Start with lower fuse_lambda

    print(f"\nUsing enhanced parameters for FID:")
    print(f"n_iters: {fid_eval_cfg['n_iters']}")
    print(f"fuse_lambda: {fid_eval_cfg['fuse_lambda']}")

    fid_scores, overall_fid = evaluate_fid_scores(
        digit_list, model_AE_list, model_dnet_list, train_loader_list,
        device, fid_eval_cfg, per_digit=True
    )

    # Uncomment these if you want to run other evaluations:
    # eval_all_digits_with_pure_noise_evolution(digit_list, eval_cfg, figure_path, model_AE_list, model_dnet_list,
    #                                          train_loader_list, test_loader_list, prob_params_list,
    #                                          per_digit=per_digit, device=device)
    #
    # compare_alg_results(digit_list,test_ds,model_AE_list,model_dnet_list,model_dae_list,model_diff_list,train_loader_list,
    #                    eval_cfg,figure_path=figure_path,index=17)