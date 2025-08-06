import torchvision.transforms as transforms
import torchvision
import torch
from architectures import MNISTConvAutoencoder, MLP, DiffusionModel
import utils
from utils import visualize_results_multi_digits, calc_mean_cov, to_np,reconstruction,calc_metrics
import matplotlib.pyplot as plt
from ldm_train import p_sample_loop
import numpy as np
import matplotlib.patches as patches
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from prettytable import PrettyTable

transform = transforms.Compose([
    transforms.ToTensor(),
])



def compare_performance(digit,test_loader,deformation,noise_level,AE_model,dnet_model,dae_model, diffusion_model, train_loader,
                        **kwargs):
    eval_cfg = kwargs.get('eval_cfg', None)
    method=eval_cfg['method']
    diffusion_steps_used=eval_cfg['diffusion_steps'] if eval_cfg else 100
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
   
   

    if deformation == 'noise':
        noisy_images = utils.add_noise(test_images,noise_factor=noise_level)
    elif deformation == 'blur':
        noisy_images = utils.add_blur(test_images,kernel=noise_level)
    elif deformation == 'SR':
        noisy_images = utils.add_down_sample(test_images,scale_factor=noise_level)
    elif deformation == 'rotate':   
        noisy_images = utils.add_rotate(test_images,min_angle=noise_level,max_angle=noise_level)
    elif deformation == 'elastic':
        noisy_images = utils.add_elastic_transform(test_images,alpha=34.0,sigma=noise_level)

    
    reconstructed_images = reconstruction(noisy_images, AE_model, dnet_model, device, method, train_loader, **kwargs)
    ours_mse,ours_psnr,ours_ssim = calc_metrics(test_images,reconstructed_images) 
   
   
    degraded_latent = dae_model.encoder(noisy_images)
    direct_reconstruction = dae_model.decoder(degraded_latent)
    dae_mse,dae_psnr,dae_ssim = calc_metrics(test_images,direct_reconstruction) 
   
   

    denoised_reconstruction, _ = p_sample_loop(
            diffusion_model,
            dae_model,
            latent_start=degraded_latent,
            diffusion_steps_used=diffusion_steps_used
        )
    diff_mse,diff_psnr,diff_ssim = calc_metrics(test_images,denoised_reconstruction) 
   
    return ours_ssim,dae_ssim,diff_ssim
    

def gradual_deformation(digit,ds,index,deformation,AE_model,dnet_model,dae_model, diffusion_model, train_loader, 
                        eval_cfg=None,figure_path=None,test_image=None,seed=None):
                      
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)     
   
    do_title=False
    #test_image = ds[index][0]
    test_image_np = utils.to_np(test_image).squeeze()

   


    
    rows = []
    diffusion_steps=eval_cfg['diffusion_steps'] if eval_cfg else 100
   
    kwargs={'eval_cfg': eval_cfg}

    if deformation == 'noise':
        #noise_list=np.linspace(0.2,1.2,6)
        noise_list=np.linspace(0.2,0.8,6)
    elif deformation == 'blur':
        noise_list=[3,5,7,9,11,13]

    elif deformation == 'SR':
        #noise_list=np.linspace(0.6,0.22,6)
        noise_list=np.linspace(0.8,0.4,6)
    elif deformation == 'rotate': 
       
        #noise_list=np.linspace(10,150,6)
        noise_list=np.linspace(5,40,6)
        
    elif deformation == 'elastic':
        #noise_list=np.linspace(3.0,1.1,6)
        noise_list=np.linspace(3.0,2.1,6)
        

    #figure = plt.figure(figsize=(7, 3))
    for i,noise_factor in enumerate(noise_list):
     
        print('diffusion_steps:',diffusion_steps)
        print('noise factor:',noise_factor)
        if deformation == 'noise':
            noisy_image = utils.add_noise(test_image,noise_factor=noise_factor)
        elif deformation == 'blur':
            noisy_image = utils.add_blur(test_image,kernel=noise_factor)
        elif deformation == 'SR':
            noisy_image = utils.add_down_sample(test_image,scale_factor=noise_factor)    
        elif deformation == 'rotate':
            noisy_image = utils.add_rotate(test_image,min_angle=noise_factor,max_angle=noise_factor)
        elif deformation == 'elastic':
            noisy_image = utils.add_elastic_transform(test_image,alpha=34.0,sigma=noise_factor)

        noisy_image = noisy_image.to(device).unsqueeze(0)
        noisy_image_np = utils.to_np(noisy_image).squeeze()

        reconstructed_images = utils.reconstruction(noisy_image, AE_model, dnet_model, device, 'dist', train_loader, **kwargs)
        reconstructed_image_np = utils.to_np(reconstructed_images).squeeze()
     

        degraded_latent = dae_model.encoder(noisy_image)
        direct_reconstruction = dae_model.decoder(degraded_latent)
        direct_reconstruction_np = utils.to_np(direct_reconstruction).squeeze()
      
        denoised_reconstruction, _ = p_sample_loop(
            diffusion_model,
            dae_model,
            latent_start=degraded_latent,
            diffusion_steps_used=diffusion_steps
        )

        denoised_reconstruction_np = utils.to_np(denoised_reconstruction).squeeze()
    
        row = {
            'degraded': noisy_image_np,
            'direct_reconstruction': direct_reconstruction_np,
            'diffusion_reconstruction': denoised_reconstruction_np,
            'dist_reconstruction': reconstructed_image_np,
            'original': test_image_np,
            
        }
        rows.append(row)

      
    if rows:
        fig, axes = plt.subplots(len(rows), 5, figsize=(16, 5 * len(rows)))
        
        # Handle case with just one row
        if len(rows) == 1:
            axes = axes.reshape(1, -1)

        for i, row in enumerate(rows):
            axes[i, 0].imshow(row['degraded'], cmap='gray')
            if i == 0 and do_title:
                axes[i, 0].set_title('Degraded',fontsize=50)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(row['direct_reconstruction'], cmap='gray')
            if i == 0 and do_title:
                axes[i, 1].set_title('direct',fontsize=50)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(row['diffusion_reconstruction'], cmap='gray')
            if i == 0 and do_title:
                axes[i, 2].set_title('diffusion',fontsize=50)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(row['dist_reconstruction'], cmap='gray')
            if i == 0 and do_title:
                axes[i, 3].set_title('ours',fontsize=50)
            axes[i, 3].axis('off')
            # Add a green rectangle around 'dist_reconstruction'
            rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='deepskyblue', linewidth=30, 
                         transform=axes[i, 3].transAxes)
            axes[i, 3].add_patch(rect)
            



            axes[i, 4].imshow(row['original'], cmap='gray')
            if i == 0 and do_title:
                axes[i, 4].set_title('original',fontsize=50)
            axes[i, 4].axis('off')

        plt.tight_layout()
        out_file = os.path.join(figure_path, f"{deformation}_digit_{digit}.png")
        plt.savefig(out_file)
        plt.close(fig)


  

def prepare_models_and_data_loaders(digit_list):

    
    latent_dim=15
    model_AE_path_list = list(range(10))
    model_dae_path_list = list(range(10))
    model_diffusion_path_list = list(range(10))
    model_AE_list = list(range(10))
    model_dnet_list = list(range(10))
    model_dae_list = list(range(10))
    model_diffusion_list = list(range(10))
    test_ds = list(range(10))
    test_loader_list = list(range(10))
    train_ds = list(range(10))
    train_loader_list = list(range(10))
    prob_params_list = list(range(10))
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)
    
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)
    for digit in digit_list:
       
        idx = (test_dataset.targets == digit)
        test_ds[digit] = torch.utils.data.Subset(test_dataset, indices=torch.where(idx)[0])
        
        test_loader_list[digit] = torch.utils.data.DataLoader(test_ds[digit], batch_size=128, shuffle=False)

      
        idx = (train_dataset.targets == digit)
        train_ds[digit] = torch.utils.data.Subset(train_dataset, indices=torch.where(idx)[0])
        
        train_loader_list[digit] = torch.utils.data.DataLoader(train_ds[digit], batch_size=128, shuffle=False)

       
        model_AE_path_list[digit] = 'Models/dist_set_digit{}.pth'.format(digit) 
        model_dae_path_list[digit] = 'Models/dae_digit{}.pth'.format(digit)
        model_diffusion_path_list[digit] = 'Models/diffusion_digit{}.pth'.format(digit)

        model_AE_list[digit] = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        model_dnet_list[digit] = MLP(latent_dim,[100,50,20]).to(device)
    
        model_dae_list[digit] = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        model_diffusion_list[digit] = DiffusionModel(latent_dim).to(device)

        print('loading {}'.format(model_AE_path_list[digit]))
        checkpoint = torch.load(model_AE_path_list[digit],weights_only=True)
        model_AE_list[digit].load_state_dict(checkpoint['AE_state_dict'])
        model_dnet_list[digit].load_state_dict(checkpoint['dnet_state_dict'])

        print('loading {}'.format(model_dae_path_list[digit]))
        checkpoint = torch.load(model_dae_path_list[digit],weights_only=True)
        model_dae_list[digit].load_state_dict(checkpoint)

        print('loading {}'.format(model_diffusion_path_list[digit]))
        checkpoint = torch.load(model_diffusion_path_list[digit],weights_only=True)
        model_diffusion_list[digit].load_state_dict(checkpoint)

        prob_params_list[digit] = calc_mean_cov(train_loader_list[digit],model_AE_list[digit])

    return model_AE_list,model_dnet_list,model_dae_list,model_diffusion_list,train_loader_list,test_loader_list,prob_params_list



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    seed=10#42
    torch.manual_seed(seed)
    np.random.seed(seed)


    digit_list = [0,1,2,3,4,5,6,7,8,9]
    #digit_list = [9]

    n_iters=7
    fuse_lambda=0.65
    delta_fuse=(0.9-fuse_lambda)/n_iters


    index=10#10#21
    latent_dim=15
    model_AE_path_list = list(range(10))
    model_dae_path_list = list(range(10))
    model_diffusion_path_list = list(range(10))
    model_AE_list = list(range(10))
    model_dnet_list = list(range(10))
    model_dae_list = list(range(10))
    model_diffusion_list = list(range(10))
    test_ds = list(range(10))
    test_loader_list = list(range(10))
    train_ds = list(range(10))
    train_loader_list = list(range(10))
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)
    
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transform,
                                           download=True)
    for digit in digit_list:

        idx = (test_dataset.targets == digit)
        test_ds[digit] = torch.utils.data.Subset(test_dataset, indices=torch.where(idx)[0])
        test_loader_list[digit] = torch.utils.data.DataLoader(test_ds[digit], batch_size=128, shuffle=False)


        idx = (train_dataset.targets == digit)
        train_ds[digit] = torch.utils.data.Subset(train_dataset, indices=torch.where(idx)[0])
        train_loader_list[digit] = torch.utils.data.DataLoader(train_ds[digit], batch_size=128, shuffle=True)

       
       
        model_AE_path_list[digit] = 'Models/dist_set_digit{}.pth'.format(digit) 
        model_dae_path_list[digit] = 'Models/dae_digit{}.pth'.format(digit)
        model_diffusion_path_list[digit] = 'Models/diffusion_digit{}.pth'.format(digit)

        model_AE_list[digit] = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        model_dnet_list[digit] = MLP(latent_dim,[100,50,20]).to(device)
    
        model_dae_list[digit] = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
        model_diffusion_list[digit] = DiffusionModel(latent_dim).to(device)

        print('loading {}'.format(model_AE_path_list[digit]))
        checkpoint = torch.load(model_AE_path_list[digit],weights_only=True)
        model_AE_list[digit].load_state_dict(checkpoint['AE_state_dict'])
        model_dnet_list[digit].load_state_dict(checkpoint['dnet_state_dict'])

        print('loading {}'.format(model_dae_path_list[digit]))
        checkpoint = torch.load(model_dae_path_list[digit],weights_only=True)
        model_dae_list[digit].load_state_dict(checkpoint)

        # print('loading {}'.format(model_ae_path_list[digit]))
        # checkpoint = torch.load(model_ae_path_list[digit],weights_only=True)
        # model_ae_list[digit].load_state_dict(checkpoint)

        print('loading {}'.format(model_diffusion_path_list[digit]))
        checkpoint = torch.load(model_diffusion_path_list[digit],weights_only=True)
        model_diffusion_list[digit].load_state_dict(checkpoint)

    diffusion_steps_vector = [500,100,100,150]
    index_list=[10,17,17,10]
    for digit in digit_list:
        for i,deformation in enumerate(['noise','SR','rotate','elastic']):
            diffusion_step = diffusion_steps_vector[i]
            index = index_list[i]
            gradual_deformation(digit,test_ds[digit],index,deformation,model_AE_list[digit],model_dnet_list[digit],
                            model_dae_list[digit],model_diffusion_list[digit],train_loader_list[digit],
                        fuse_lambda=fuse_lambda,sigma=1,delta_fuse=delta_fuse,n_iters=n_iters,diffusion_steps=diffusion_step)


def main2():

    seed=10
    torch.manual_seed(seed)
    np.random.seed(seed)
    digit_list = [0,1,2,3,4,5,6,7,8,9]
    digit_list=[2]
   
    per_digit=True
    method = 'dist'
    sigma=1
    n_iters=2
    fuse_lambda=0.65
    delta_fuse=0.05
    prob_params_list = list(range(10))
   
    n_iters=9
    fuse_lambda=0.75#0.65
    delta_fuse=(0.9-fuse_lambda)/n_iters


    model_AE_list,model_dnet_list,_,_,train_loader_list,test_loader_list,prob_params_list = prepare_models_and_data_loaders(digit_list)

    for digit in digit_list:
        if not per_digit:
            print('visualizing results for digit {}'.format(digit))
            file_path = 'Figures/paper_alldigit_digit{}'.format(digit)
            visualize_results_multi_digits(
                                            digit_list, model_AE_list, model_dnet_list, test_loader_list[digit], device, 
                                            method=method,num_images=10,train_loader_list=train_loader_list,
                                            fuse_lambda=fuse_lambda,sigma=sigma,delta_fuse=delta_fuse,
                                            n_iters=n_iters,file_path=file_path,
                                            prob_params=prob_params_list[digit],digit=digit,offset=50,edge_color='midnightblue'
                                            )
        else:
           
            file_path = 'Figures/paper_perdigit_digit{}'.format(digit)
            visualize_results_multi_digits(
                                            [digit], model_AE_list, model_dnet_list, test_loader_list[digit], device, 
                                            method=method,num_images=10,train_loader_list=train_loader_list,
                                            fuse_lambda=fuse_lambda,sigma=sigma,delta_fuse=delta_fuse,
                                            n_iters=n_iters, file_path=file_path,
                                            prob_params=prob_params_list[digit],digit=digit,offset=50,edge_color='darkblue'
                                            )


def main3():

    seed=10
    torch.manual_seed(seed)
    np.random.seed(seed)
    digit_list = [0,1,2,3,4,5,6,7,8,9]
    digit_list=[2]
    
    sigma=1
    n_iters=3
    fuse_lambda=0.65
    delta_fuse=0.05
   
    deformation_list = ['rotate','elastic','SR','noise']


    model_AE_list,model_dnet_list,model_dae_list,model_diffusion_list,train_loader_list,test_loader_list,prob_params_list = prepare_models_and_data_loaders(digit_list)
    for digit in digit_list:
        print('metric calculation for digit {}'.format(digit))
        utils.calculate_digit_metrics(test_loader_list[digit],train_loader_list[digit],model_AE_list[digit],model_dnet_list[digit],
                                      model_dae_list[digit],model_diffusion_list[digit],device,deformation_list=deformation_list,
                                      fuse_lambda=fuse_lambda,sigma=sigma,delta_fuse=delta_fuse,
                                            n_iters=n_iters,diffusion_steps=100)





if __name__ == '__main__':
    main2()