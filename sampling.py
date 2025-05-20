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

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, dit_model_config, diffusion_config, dataset_config,batch_num):

    latent_height = dit_model_config["img_height"]
    latent_width = dit_model_config["img_width"]

    # We start with white noise with dimension same as the latent representation
    xt = torch.randn((train_config['num_samples'],
                      dit_model_config['latent_channels'],
                      latent_height,
                      latent_width)).to(device)

    # Using VAE for decoding from stable diffusion

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)

    vae.eval()

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):


        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Reverse process to get the previous timestep xt from the current xt
        xt_prev, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        if i != 0:

            xt = xt_prev  # Final clean image at the end of reverse diffusion
            
        else:

            ims = xt_prev

            with torch.no_grad():

                ims = vae.to(device).decode(ims).sample  # Convertno latent space to image space

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

            grid = make_grid(ims, nrow=int(np.ceil(np.sqrt(ims.shape[0]))), normalize=True)
            img = torchvision.transforms.ToPILImage()(grid)


            if not os.path.exists(join(args.save_root_path, 'samples',dataset_config["dataset_name"])):
                os.mkdir(join(args.save_root_path, 'samples',dataset_config["dataset_name"]))

            if i == 0:
                 # For saving the generated images
                for j, img_ in enumerate(ims):
                    save_image(img_, join(args.save_root_path, 'samples',dataset_config["dataset_name"],f'{save_tag}_x0_{batch_num*10 + j + 1}.png'))

def infer(args,batch_num):
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

    # Get latent image size

    model = DiT(dit_model_config).to(device)

    model.eval()


    save_tag = f"{dataset_config['dataset_name']} {float(train_config['learning_rate'])} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']} {dit_model_config['patch_height']}".replace(" ", "_")

    print(save_tag)

    assert os.path.exists(join(args.save_root_path, 'models',save_tag + '_dit_model.pth'))

     
    
    model.load_state_dict(torch.load(join(args.save_root_path,'models', save_tag + '_dit_model.pth'),
                                     map_location=device))
    print('Loaded dit checkpoint')



    with torch.no_grad():
        sample(model, scheduler, train_config, dit_model_config, diffusion_config, dataset_config,batch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dit image generation')
    parser.add_argument('--config_file',default= '/home/sid/DiT/config/cifar10.yaml', type=str)
    parser.add_argument('--save_root_path',default= '/home/sid/DiT/results/', type=str)
    args = parser.parse_args()

    start = time.time()
    for batch_num in range(1000):  # THis is for 10K samples. Running a batch of 10 samples at a time.
        infer(args,batch_num)

    print("Total Time Taken: ", time.time() - start)

















               # img.save(join(args.save_root_path, 'samples',dataset_config["dataset_name"], 'x0_{}.png'.format(i)))
                # img.close()
                # show_image_grid(ims,save_path = join(args.save_root_path, 'samples',dataset_config["dataset_name"], 'x0_grid_{}.png'.format(i)))

    # Create output directories
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