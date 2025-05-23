import torch
import torchvision
import argparse
import yaml
import os
from os.path import join
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from model import DiT
from DDPM_Sampler import DDPMSampler
from diffusers import AutoencoderKL
from utils import show_image_grid

import numpy as np
from torchvision.utils import save_image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, dit_model_config, diffusion_config, dataset_config, class_labels_list = [11,21,31,41,45]):

    latent_height = dit_model_config["img_height"]
    latent_width = dit_model_config["img_width"]

    class_labels = [int(label) for label in class_labels_list]

    n = len(class_labels)

    xt = torch.randn(n, dit_model_config['latent_channels'], latent_height, latent_width, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    xt = torch.cat([xt, xt], 0)
    y_null = torch.tensor([1000] * n, device=device)  # since in imagenet there are classes from 0 to 999
    y = torch.cat([y, y_null], 0)


    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):


        noise_pred,_,_ = model.forward_with_cfg(xt, torch.as_tensor(i).unsqueeze(0).to(device),y,dit_model_config['cfg_weight'])

        # print(noise_pred.shape)

        # Use scheduler to get x0 and xt-1
        xt_prev, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        if i != 0:

            xt = xt_prev
            
        else:
            
            print(" Getting the final Clean Image ")
            ims = xt_prev

            ims = vae.to(device).decode(ims).sample

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

            grid = make_grid(ims, nrow=int(np.ceil(np.sqrt(ims.shape[0]))), normalize=True)
            img = torchvision.transforms.ToPILImage()(grid)

            # print(img.size)
            print("saving the image")

            if not os.path.exists(join(args.save_root_path, 'samples',dataset_config["dataset_name"])):
                os.mkdir(join(args.save_root_path, 'samples',dataset_config["dataset_name"]))

            # if i == 0:

            save_tag = f"{dataset_config['dataset_name']} {float(train_config['learning_rate'])} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']} {dit_model_config['patch_height']}".replace(" ", "_")
            
            show_image_grid(ims.chunk(2,dim=0)[0],save_path = join(args.save_root_path, 'samples',dataset_config["dataset_name"], save_tag + '_x0_grid_{}.png'.format(i)))

            # for j, img_ in enumerate(ims):
            #     save_image(img_, join(args.save_root_path, 'samples',dataset_config["dataset_name"],f'{save_tag}_x0_{j}.png'))

    print("Inference complete!!!")


def infer(args):
    # Read the config file #
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_forward_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_model_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = DDPMSampler(diffusion_config)

    # Initializing the pretrained model for inference\

    model = DiT(dit_model_config).to(device)

    model.eval()


    save_tag = f"{dataset_config['dataset_name']} {float(train_config['learning_rate'])} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']} {dit_model_config['patch_height']}".replace(" ", "_")

    # print(save_tag)

    # print(join(args.save_root_path, 'models',save_tag + '_dit_model.pth'))

    assert os.path.exists(join(args.save_root_path, 'models',save_tag + '_dit_model.pth'))

     
    # Loading the pretrained checkpoint 

    model.load_state_dict(torch.load(join(args.save_root_path,'models', save_tag + '_dit_model.pth'),
                                     map_location=device))
    print('Loaded dit checkpoint')


    # The reverse sampling process to get the clean image
    with torch.no_grad():
        sample(model, scheduler, train_config, dit_model_config, diffusion_config, dataset_config,args.class_labels_list.split(','))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dit image generation')
    parser.add_argument('--config_file',default= '/home/rohit/DiT/config/cifar10.yaml', type=str)
    parser.add_argument('--save_root_path',default= '/home/rohit/DiT/results/', type=str)
    parser.add_argument('--class_labels_list',default= '11,21,31,41,45,51,61,71,81', type=str) # comma separated list of class labels

    args = parser.parse_args()


    print("Class labels list are :", args.class_labels_list.split(","))


    infer(args)

















    #            img.save(join(args.save_root_path, 'samples',dataset_config["dataset_name"], 'x0_{}.png'.format(i)))
    #             img.close()
    #             show_image_grid(ims,save_path = join(args.save_root_path, 'samples',dataset_config["dataset_name"], 'x0_grid_{}.png'.format(i)))


    # # Create output directories
    # if not os.path.exists(train_config['task_name']):
    #     os.mkdir(train_config['task_name'])

    # vae = VAE(im_channels=dataset_config['im_channels'],
    #           model_config=autoencoder_model_config)
    # vae.eval()

    # Load vae if found
    # assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
    #     "VAE checkpoint not present. Train VAE first."
    # vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                             train_config['vae_autoencoder_ckpt_name']),
    #                                map_location=device), strict=True)
    # print('Loaded vae checkpoint')

    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)