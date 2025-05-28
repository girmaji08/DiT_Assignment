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
from diffusers import AutoencoderKL
from utils import show_image_grid

import numpy as np
from torchvision.utils import save_image

import time


parser = argparse.ArgumentParser(description='Arguments for dit image generation')
parser.add_argument('--config_file',default= '/home/sid/DiT/config/cifar10.yaml', type=str)
parser.add_argument('--save_root_path',default= '/home/sid/DiT/results/', type=str)
args = parser.parse_args()

start = time.time()



class DDIMSampler:

    def __init__(self,args):
        
        num_timesteps = args["num_timesteps"]
        beta_start = args["beta_start"]
        beta_end = args["beta_end"]
        self.n_sampling_steps = args["ddim_n_sampling_steps"]
        self.eta = args["eta"]

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        self.sampling_timesteps = torch.linspace(0, num_timesteps - 1, self.n_sampling_steps, dtype=torch.long)

    def sample(self,model,
               train_config,
               dit_model_config,
               diffusion_config,
               dataset_config, device, batch_num = 1):

        latent_height = dit_model_config["img_height"]
        latent_width = dit_model_config["img_width"]
        # We start with white noise with dimension same as the latent representation


        model.eval()


            # Using VAE for decoding from stable diffusion
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)
        vae.eval()

        with torch.no_grad():
        
            x = torch.randn((train_config['num_samples'],
                            dit_model_config['latent_channels'],
                            latent_height,
                            latent_width)).to(device)

        x_intermediates = []

        for i in tqdm(reversed(range(self.n_sampling_steps))):
            timestep = self.sampling_timesteps[i].to(device)

            timestep_prev = self.sampling_timesteps[i - 1] if i > 0 else torch.tensor([-1])

            t_batch = torch.full((x.shape[0],), timestep.item(), device=x.device, dtype=torch.long)

            predicted_noise = model(x, t_batch)


            alpha_cum_prod_t = self.alpha_cum_prod[timestep].to(device)
            alpha_cum_prod_t_prev = self.alpha_cum_prod[timestep_prev].to(device) if i > 0 else torch.tensor([1.0]).to(device)

            x_0_predicted = (x - torch.sqrt(1 - alpha_cum_prod_t).view(-1,1,1,1) * predicted_noise) / torch.sqrt(alpha_cum_prod_t).view(-1,1,1,1)


            sigma_t = self.eta * torch.sqrt(((1-alpha_cum_prod_t_prev) / (1 - alpha_cum_prod_t)) * (1 - (alpha_cum_prod_t / alpha_cum_prod_t_prev)))


            c1 = torch.sqrt(alpha_cum_prod_t_prev).view(-1, 1, 1, 1)
            c2 = torch.sqrt(1 - alpha_cum_prod_t_prev - sigma_t ** 2).view(-1, 1, 1, 1)

            if i > 0:
                
                noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)

                x = c1 * x_0_predicted  + c2 * predicted_noise + sigma_t.view(-1, 1, 1, 1) * noise

                x_intermediates.append(x)

            else:
                x = x_0_predicted
                x_intermediates.append(x)


        return x,torch.stack(x_intermediates).permute(1,0,2,3,4)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


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


# Get latent image size

model = DiT(dit_model_config).to(device)

model.eval()


save_tag = f"{dataset_config['dataset_name']} {float(train_config['learning_rate'])} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']} {dit_model_config['patch_height']}".replace(" ", "_")

print(save_tag)

assert os.path.exists(join(args.save_root_path, 'models',save_tag + '_dit_model.pth'))

    

model.load_state_dict(torch.load(join(args.save_root_path,'models', save_tag + '_dit_model.pth'),
                                    map_location=device))
print('Loaded dit checkpoint')


sampler = DDIMSampler(diffusion_config)

with torch.no_grad():
    print("Sampling from model...")
    x0_final,x_intermediates = sampler.sample(model,train_config,dit_model_config,diffusion_config,dataset_config,device)

    print("Sampling complete!!! in latent space")



print("Decoding to image space...")


vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)
vae.eval()

ims = vae.to(device).decode(x0_final).sample  # Convertno latent space to image space


ims = torch.clamp(ims, -1., 1.).detach().cpu()

ims = (ims + 1) / 2 


grid = make_grid(ims, nrow=int(np.ceil(np.sqrt(ims.shape[0]))), padding=2, normalize=False)
img = torchvision.transforms.ToPILImage()(grid)


ims_uint8 = (ims * 255).clamp(0, 255).to(torch.uint8).detach().cpu()


if not os.path.exists(join(args.save_root_path, 'samples',dataset_config["dataset_name"])):
    os.mkdir(join(args.save_root_path, 'samples',dataset_config["dataset_name"]))


output_dir = join(args.save_root_path, 'samples', dataset_config["dataset_name"])
os.makedirs(output_dir, exist_ok=True)


for j, img_ in enumerate(ims_uint8):

    save_path = join(output_dir, f'{save_tag}_x0_{j + 1}.png')
    torchvision.utils.save_image(img_.float() / 255.0, save_path)

# save all images in a grid

grid_save_path = join(output_dir, f'{save_tag}_x0_grid.png')
torchvision.utils.save_image(grid, grid_save_path)

print(" Sampling complete and individual final samples saved!!!")


assert x_intermediates.shape[1] == diffusion_config['ddim_n_sampling_steps'], "Number of intermediates does not match the number of sampling steps"

print("Saving intermediate samples...")


for i,x_inter in enumerate(x_intermediates):


    decoded_intermediates = []

    for chunk in torch.split(x_inter, 10, dim=0):
        x_inter_chunk = vae.to(device).decode(chunk).sample
        x_inter_chunk = torch.clamp(x_inter_chunk, -1., 1.).detach().cpu()
        x_inter_chunk = (x_inter_chunk + 1) / 2
        decoded_intermediates.append(x_inter_chunk)

    x_inter = torch.cat(decoded_intermediates, dim=0)

    grid = make_grid(x_inter, nrow=int(np.ceil(np.sqrt(x_inter.shape[0]))), padding=2, normalize=False)
    img = torchvision.transforms.ToPILImage()(grid)

    grid_save_path = join(output_dir, f'{save_tag}_intermediate_grid_sample_{i+1}.png')
    torchvision.utils.save_image(grid, grid_save_path)


