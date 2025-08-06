import os
import numpy as np
import torch,torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from architectures import MNISTConvAutoencoder, MLP, DiffusionModel
from losses import loss_dist_al
from train import train_al,train_dae_diffusion_model
from utils import visualize_results_multi_digits,calc_mean_cov,to_np
from mnist_eval import gradual_deformation,compare_performance
from prettytable import PrettyTable
from utils import add_noise, add_down_sample,add_elastic_transform,reconstruction,add_rotate,calc_metrics,visualize_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

transform = transforms.Compose([
    transforms.ToTensor(),
])

SR_SEVERE=0.22
ROT_SEVERE=-150
EL_SEVERE=1.1
NOISE_SEVERE=1.2#0.8#1.3

SR_MILD=0.55
ROT_MILD=30
EL_MILD=2.4
NOISE_MILD=0.7 #0.75

SR_INTER=(SR_MILD+SR_SEVERE)/2
ROT_INTER=(ROT_MILD+ROT_SEVERE)/2
EL_INTER=(EL_MILD+EL_SEVERE)/2
NOISE_INTER=(NOISE_MILD+NOISE_SEVERE)/2


def prepare_data(cfg):

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

    train_ds = train_dataset
    test_ds = test_dataset
   

    latent_dim = cfg['latent_dim']
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
       
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)
    
    model_AE = MNISTConvAutoencoder(latent_dim).to(device)
    model_dnet = MLP(latent_dim,cfg['mlp_arch']).to(device)

    model_dae = MNISTConvAutoencoder(latent_dim=latent_dim).to(device)
    model_diffusion = DiffusionModel(latent_dim).to(device)
       
    
    return train_loader, test_loader, model_AE, model_dnet, model_dae, \
            model_diffusion,train_ds, test_ds


def train_all(cfg,AE_model,dnet_model,train_loader,model_path,figure_path):

    
    
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
                        lam3=cfg['lam3'],lam6=cfg['lam6'],lam7=cfg['lam7'],do_update=cfg['do_update']) 
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
    plt.figure()
    lam_vec = np.array(lam_list)
    plt.plot(lam_vec, label='lam3')
        
    plt.title(f'lambda, loss3={loss_vector[2]:.3f} loss4={loss_vector[3]:.3f}')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(figure_path,f'train_lam.png'))
    plt.close('all')   

def reload_models(model_path,diff_model_path,dae_model_path,AE_model,dnet_model,diff_model,dae_model):
   
    checkpoint = torch.load(model_path,weights_only=True)
    AE_model.load_state_dict(checkpoint['AE_state_dict'])
    dnet_model.load_state_dict(checkpoint['dnet_state_dict'])
    dae_model.load_state_dict(torch.load(dae_model_path,weights_only=True))
    diff_model.load_state_dict(torch.load(diff_model_path,weights_only=True))
   
    
    return AE_model, dnet_model, diff_model, dae_model

def compare_alg_results(test_ds,AE_model,dnet_model,dae_model,diffusion_model,train_loader,
                       eval_cfg,figure_path=None,index=10,seed=42):
     
   
       
    test_image = test_ds[index][0]
    for i,deformation in enumerate(['noise','SR','rotate','elastic']):
        
        gradual_deformation(-1,test_ds,None,deformation,AE_model,dnet_model,
                        dae_model,diffusion_model,train_loader,
                    eval_cfg,figure_path,test_image=test_image,seed=seed)
   
def eval_all_digits(eval_cfg,figure_path,AE_model,dnet_model,train_loader,test_loader,offset_idx=0,seed=42):

    # dnet_model.eval()   
    # AE_model.eval()   

    torch.manual_seed(seed)
    np.random.seed(seed)

    method = eval_cfg['method']
    n_iters = eval_cfg['n_iters']
    fuse_lambda = eval_cfg['fuse_lambda']
    sigma = eval_cfg['sigma']
    delta_fuse = eval_cfg['delta_fuse']


    path_noise1 = os.path.join(figure_path,'output_all_noise1.png')
    path_noise2 = os.path.join(figure_path,'output_all_noise2.png')
    path_SR1 = os.path.join(figure_path,'output_all_SR1.png')
    path_SR2 = os.path.join(figure_path,'output_all_SR2.png')
    path_elastic1 = os.path.join(figure_path,'output_all_elastic1.png')
    path_elastic2 = os.path.join(figure_path,'output_all_elastic2.png')
    path_rotate1 = os.path.join(figure_path,'output_all_rotate1.png')
    path_rotate2 = os.path.join(figure_path,'output_all_rotate2.png')

    images, _ = next(iter(test_loader))
    images = images.to(device)


    print(f'testing on noisy images 0.6')
    noisy_images = add_noise(images,noise_factor=eval_cfg['noise_factor']).to(device)
    print_results(images,noisy_images, AE_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,
                                              path_noise1,path_noise2,eval_cfg=eval_cfg)

    print(f'testing on downsampled images 0.5')
    noisy_images = add_down_sample(images,scale_factor=eval_cfg['sr_scale']).to(device)
    print_results(images,noisy_images, AE_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,
                                              path_SR1,path_SR2,eval_cfg=eval_cfg)
        
    print(f'testing on elastic transformed images 30,1.9')
    noisy_images = add_elastic_transform(images, alpha=eval_cfg['elastic_alpha'],sigma=eval_cfg['elastic_sigma']).to(device)
    print_results(images,noisy_images, AE_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,
                                              path_elastic1,path_elastic2,eval_cfg=eval_cfg)

    print(f'testing on rotated images -10,-30')
    noisy_images = add_rotate(images,min_angle=eval_cfg['min_angle'],max_angle=eval_cfg['max_angle']).to(device)
    print_results(images,noisy_images, AE_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset_idx,offset_idx+60,
                                              path_rotate1,path_rotate2,eval_cfg=eval_cfg)
    
    


def print_results(images,noisy_images, dae_model, dnet_model, device, method, train_loader, 
                                              fuse_lambda,sigma,delta_fuse,n_iters,offset1,offset2,output_path1,output_path2,eval_cfg=None):

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

def qualitative_results(test_loader,model_AE ,model_dnet,model_dae,model_diff,train_loader,
                       eval_cfg,type='severe',seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    

    ours_ssim_sr_list=[]
    diff_ssim_sr_list=[]
    dae_ssim_sr_list=[]
    ours_ssim_el_list=[]
    diff_ssim_el_list=[]
    dae_ssim_el_list=[]
    ours_ssim_rot_list=[]
    diff_ssim_rot_list=[]
    dae_ssim_rot_list=[]
    ours_ssim_noi_list=[]
    diff_ssim_noi_list=[]
    dae_ssim_noi_list=[]

    #severe
    if type=='severe':
        sr_level=SR_SEVERE
        rot_level=ROT_SEVERE
        el_level=EL_SEVERE
        noi_level=NOISE_SEVERE
       

    #intermediate
    if type=='intermediate':
        sr_level=SR_INTER
        rot_level=ROT_INTER
        el_level=EL_INTER
        noi_level=NOISE_INTER
        

    if type=='mild':   
        sr_level=SR_MILD
        rot_level=ROT_MILD
        el_level=EL_MILD
        noi_level=NOISE_MILD


   
        #print('evaluating model for digit {}'.format(digit))
    table = PrettyTable()
    table.field_names = ["Method", "noise", "rotate", "elastic", "SR"]

    
    ours_ssim_sr,dae_ssim_sr,diff_ssim_sr = compare_performance(-1,test_loader,'SR',sr_level,model_AE,model_dnet,
                        model_dae, model_diff, train_loader,eval_cfg=eval_cfg)
    
    
    ours_ssim_rot,dae_ssim_rot,diff_ssim_rot=compare_performance(-1,test_loader,'rotate',rot_level,model_AE,model_dnet,
                        model_dae, model_diff, train_loader,eval_cfg=eval_cfg)
    
    ours_ssim_el,dae_ssim_el,diff_ssim_el=compare_performance(-1,test_loader,'elastic',el_level,model_AE,model_dnet,
                        model_dae, model_diff, train_loader,eval_cfg=eval_cfg)
    
    ours_ssim_noi,dae_ssim_noi,diff_ssim_noi=compare_performance(-1,test_loader,'noise',noi_level,model_AE,model_dnet,
                        model_dae, model_diff, train_loader,eval_cfg=eval_cfg)

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
                   f'{np.min(dae_ssim_el_list):.2f}', f'{np.min(dae_ssim_sr_list):.2f}'])
    table.add_row(["Diffusion", f'{np.mean(diff_ssim_noi_list):.2f}', f'{np.mean(diff_ssim_rot_list):.2f}',
                    f'{np.min(diff_ssim_el_list):.2f}', f'{np.min(diff_ssim_sr_list):.2f}'])
    table.add_row(["Ours", f'{np.mean(ours_ssim_noi_list):.2f}', f'{np.mean(ours_ssim_rot_list):.2f}',
                    f'{np.min(ours_ssim_el_list):.2f}', f'{np.min(ours_ssim_sr_list):.2f}'])
       
    print(table) 

def generate_helix_dataset(num_samples=1000, noise_std=0.05, radius=1, pitch=0.1):
    t = np.linspace(0, 4 * np.pi, num_samples)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t
    
    # Optional: Add Gaussian noise
    x += np.random.normal(0, noise_std, size=num_samples)
    y += np.random.normal(0, noise_std, size=num_samples)
    z += np.random.normal(0, noise_std, size=num_samples)
    
    return np.stack([x, y, z], axis=1)

if __name__ == "__main__":

   
    
    
    retrain = True
    retrain = False
    retrain_diffusion = True
    retrain_diffusion = False
    lam3=0
    lam6=1
    lam7=1
    latent=20

    per_digit=False

    train_cfg = {}
    eval_cfg = {}
    train_diff_cfg={}

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_name='Experiments/MNIST_all_digit_without_eikonal'
    os.makedirs(experiment_name, exist_ok=True)
    model_path = os.path.join(experiment_name,'Models')
    figure_path = os.path.join(experiment_name,'Figures')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    
    
    if lam3 == 0:
        do_update = False
    else:
        do_update = True
    
  
    AE_model_path = os.path.join(model_path,f'dist_set_lam3_{lam3}_lam6_{lam6}_lam7_{lam7}_lat{latent}.pth')
    #AE_model_path = os.path.join(model_path,'dist_set_all_digits_noeik_sig.pth')
    diff_model_path = os.path.join(model_path,f'diff_lat{latent}.pth')
    dae_model_path = os.path.join(model_path,f'dae_lat{latent}.pth')


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
    train_cfg['epochs'] = 50 #20
    train_cfg['mlp_arch']=[120,50,20]
    train_cfg['experiment_name'] = experiment_name
    train_cfg['loss_func']=loss_dist_al
   
    train_diff_cfg['min_noise'] = train_cfg['min_noise']
    train_diff_cfg['max_noise'] = train_cfg['max_noise']
    train_diff_cfg['dae_epochs'] = 20 #100
    train_diff_cfg['diff_epochs'] = 50 # 50#100
    train_diff_cfg['lr'] = 1e-3
    train_diff_cfg['latent_dim'] = latent
    train_diff_cfg['diffusion_steps'] = 200#1000#1200
   

   
    eval_cfg['method'] = 'dist'
    eval_cfg['n_iters'] = 3
    eval_cfg['fuse_lambda'] = 0.75#0.75
    eval_cfg['delta_fuse'] = (0.9-eval_cfg['fuse_lambda'])/eval_cfg['n_iters']
    eval_cfg['sigma'] = 1
    eval_cfg['noise_factor'] = 0.6#NOISE_MILD#0.4
    eval_cfg['elastic_alpha']=34.0#34.0 #34.0
    eval_cfg['elastic_sigma']=2.4 #EL_MILD#1.5 #1.25
    eval_cfg['sr_scale']=0.55#SR_MILD#0.35
    eval_cfg['min_angle']=30#ROT_MILD#-80
    eval_cfg['max_angle']=30#ROT_MILD#-80
    eval_cfg['num_images'] = 10
    eval_cfg['offset'] = 15#30
    eval_cfg['diffusion_steps']=200#200#150 
    eval_cfg['compare_index'] = 150#150 #500

    train_loader, test_loader, AE_model,\
    dnet_model, dae_model, diff_model, train_ds, test_ds = prepare_data(train_cfg)
    
    if retrain_diffusion:
        train_dae_diffusion_model(-1, train_loader, dae_model, diff_model, device, 
                              train_diff_cfg['dae_epochs'], train_diff_cfg['diff_epochs'],lr= train_diff_cfg['lr'], 
                              diffusion_steps=train_diff_cfg['diffusion_steps'],
                              dae_model_path=dae_model_path,diff_model_path= diff_model_path,min_noise=train_diff_cfg['min_noise'],
                             max_noise=train_diff_cfg['max_noise'])
    
    if retrain:
        train_all(train_cfg,AE_model,dnet_model,train_loader,AE_model_path,figure_path)
       
    
   
    AE_model, dnet_model,diff_model,dae_model=reload_models(AE_model_path,diff_model_path,dae_model_path,AE_model,dnet_model,diff_model,dae_model)

    AE_model.eval()
    dnet_model.eval()
    diff_model.eval()
    dae_model.eval()
    for param in AE_model.parameters():
        param.requires_grad = False
    for param in diff_model.parameters():
        param.requires_grad = False 
    for param in dae_model.parameters():
        param.requires_grad = False          
   
    eval_all_digits(eval_cfg,figure_path,AE_model,dnet_model,train_loader,test_loader,offset_idx=eval_cfg['offset'] )
   

    compare_alg_results(test_ds,AE_model,dnet_model,dae_model,diff_model,train_loader,
                       eval_cfg,figure_path=figure_path,index=eval_cfg['compare_index'],seed=seed)
    
    print('mild')
    qualitative_results(test_loader,AE_model,dnet_model,dae_model,diff_model,train_loader,
                       eval_cfg,type='mild')
    
    print('intermediate')
    qualitative_results(test_loader,AE_model,dnet_model,dae_model,diff_model,train_loader,
                       eval_cfg,type='intermediate')
    
    print('severe')
    qualitative_results(test_loader,AE_model,dnet_model,dae_model,diff_model,train_loader,
                       eval_cfg,type='severe')
    
   