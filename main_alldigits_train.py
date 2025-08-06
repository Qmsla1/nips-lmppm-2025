import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import os
import matplotlib.pyplot as plt
import warnings



from losses import loss_L2, loss_prob, loss_dist, loss_dist_al
from train import train,train_al
from architectures import MNISTConvAutoencoder,MNISTConvDnet, MLP
from utils import calc_mean_cov, generate_images_from_prob, visualize_results, to_tens, visualize_results_multi_digits
from utils import add_noise, add_down_sample,add_elastic_transform,reconstruction,add_rotate,calc_metrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import umap
# Suppress various warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

# Set random seed for reproducibility

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
    plt.close()


def load_omniglot_data(batch_size=128,digit=None):
   
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor()  # Convert to tensor (1, 28, 28)
])
    # Load the Omniglot dataset
    train_dataset = torchvision.datasets.Omniglot(
        root='./data', 
        background=True,  # Use background set for training
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.Omniglot(
        root='./data', 
        background=False,  # Use evaluation set for testing
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, test_loader
def load_mnist_data(batch_size=128):

    transform = transforms.Compose([
    transforms.ToTensor(),])
    # Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
def print_results(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset1,offset2,output_path1,output_path2):


    eval_cfg={}
    eval_cfg['n_iters'] = n_iters
    eval_cfg['fuse_lambda'] = fuse_lambda
    eval_cfg['delta_fuse'] = delta_fuse
    eval_cfg['sigma'] = sigma
    eval_cfg['method'] = method

    denoised_images = reconstruction(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda=fuse_lambda,sigma=sigma,delta_fuse=delta_fuse,
                                            n_iters=n_iters,eval_cfg=eval_cfg)

    images_np = images.cpu().numpy().squeeze()
    denoised_image_np =  denoised_images.cpu().detach().numpy().squeeze()
    noisy_image_np = noisy_images.cpu().numpy().squeeze()

    mse_ours,psnr_ours,ssim_ours = calc_metrics(noisy_images,denoised_images)
    print(f'Ours: MSE: {mse_ours:.2f} PSNR: {psnr_ours:.2f}, SSIM: {ssim_ours:.2f}')   

    visualize_images(images_np, noisy_image_np, denoised_image_np, offset_idx=offset1,num_images=20,output_path=output_path1)
    visualize_images(images_np, noisy_image_np, denoised_image_np, offset_idx=offset2,num_images=20,output_path=output_path2)

    pass

if __name__ == "__main__":


   

    
    torch.manual_seed(42)
    np.random.seed(42)


    do_mnist = False
    retrain = False
    lr_ae = 1e-3
    lr_dnet = 1e-5
    loss_func=loss_dist_al
    epochs=20
   
   
    sigma=1
    method = 'dist'
    n_iters=2
    fuse_lambda=0.7
    delta_fuse=(0.9-fuse_lambda)/n_iters
    offset_idx=0
   
    if do_mnist:
        print('loading mnist data')
        model_AE_path = 'Models/dist_set_all_digits_noeik_sig.pth'
        latent_dim=20
        eps=1e-10
        train_loader,test_loader = load_mnist_data(batch_size=128)
        dae_model= MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        dnet_model = MLP(latent_dim,[120,50,20]).to(device)
        n_iters=3#2
        fuse_lambda=0.8#0.8
        delta_fuse=(0.9-fuse_lambda)/n_iters
        all_images=[]
        for images, _ in train_loader:
            all_images.append(images)

        # Concatenate all batches along the first dimension (batch dimension)
        all_images = torch.cat(all_images, dim=0)
        all_images = all_images.to(device)
     
    else:
        
        print('loading omniglot data')
        
        model_AE_path = 'Models/dist_set_omniglot.pth'
        latent_dim=64
        eps=1e-10
        train_loader,test_loader = load_omniglot_data(batch_size=128)
        dae_model= MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        #dnet_model = MLP(latent_dim,[200,100,50,20]).to(device)
        dnet_model = MLP(latent_dim,[300,200,100,50]).to(device)
        epochs=80
        n_iters=2
        fuse_lambda=0.8
        delta_fuse=(0.9-fuse_lambda)/n_iters
        all_images=[]
        for images, _ in train_loader:
            all_images.append(images)

        # Concatenate all batches along the first dimension (batch dimension)
        all_images = torch.cat(all_images, dim=0)
        all_images = all_images.to(device)



    if retrain:
        print('training model from scratch')
        dae_model.train()
        dnet_model.train()
        
        optimizer = optim.Adam([
            {'params': dae_model.parameters(), 'lr': lr_ae},
            {'params': dnet_model.parameters(), 'lr': lr_dnet}
            ])
       
        train_losses,lam_list = train_al(dae_model, train_loader, loss_func, optimizer, 
                            device, epochs=epochs, dnet=dnet_model, eps=eps, do_update=False, all_images=all_images) 
        checkpoint = {
            'AE_state_dict': dae_model.state_dict(),
            'dnet_state_dict': dnet_model.state_dict()}
        print('saving {}'.format(model_AE_path))
        torch.save(checkpoint, model_AE_path )
        plt.figure()
        plt.plot(train_losses)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('train_loss_alldigits.png')
        plt.figure()
        lam_vec = np.array(lam_list)
        plt.plot(lam_vec,label='lam3')
        plt.title('lambda')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig('train_lam_alldigit.png')

    else:
        print('loading {}'.format(model_AE_path))
        checkpoint = torch.load(model_AE_path,weights_only=True)
        dae_model.load_state_dict(checkpoint['AE_state_dict'])
        dnet_model.load_state_dict(checkpoint['dnet_state_dict'])

    dnet_model.eval()   
    dae_model.eval()   

    images, _ = next(iter(test_loader))
    images = images.to(device)
    # noisy_images = add_noise(images,noise_factor=0.4).to(device)
    # noisy_images = add_down_sample(images,scale_factor=0.7).to(device)
    # noisy_images = add_elastic_transform(images, alpha=10.0,sigma=1.0).to(device)
    #noisy_images = add_elastic_transform(images, alpha=30.0,sigma=1.8).to(device)
    #noisy_images = add_rotate(images,min_angle=-10,max_angle=-30).to(device)
   

    print(f'testing on noisy images 0.6')
    noisy_images = add_noise(images,noise_factor=0.6).to(device)
    print_results(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,'Figures/output_all_noise_1.png','Figures/output_all_noise_2.png')

    print(f'testing on downsampled images 0.5')
    noisy_images = add_down_sample(images,scale_factor=0.5).to(device)
    print_results(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,'Figures/output_all_SR_1.png','Figures/output_all_SR_2.png')
        
    print(f'testing on elastic transformed images 30,1.9')
    noisy_images = add_elastic_transform(images, alpha=30.0,sigma=1.9).to(device)
    print_results(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,'Figures/output_all_elastic_1.png','Figures/output_all_elastic_2.png')

    print(f'testing on rotated images -10,-30')
    noisy_images = add_rotate(images,min_angle=-10,max_angle=-30).to(device)
    print_results(noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,'Figures/output_all_rot_1.png','Figures/output_all_rot_2.png')
    