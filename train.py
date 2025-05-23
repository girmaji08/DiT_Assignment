# Python Script for training the Simple Diffusion Transformer Model

import torch
import yaml
import argparse
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from dataset import LandscapesDataset,ImageNetMiniDataset
from torch.utils.data import DataLoader
from model import DiT

from DDPM_Sampler import DDPMSampler

import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import glob
from utils import load_latents_imagenet_mini


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):

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


    if dataset_config['dataset_name'] == 'landscapehq':
        im_dataset = LandscapesDataset(dataset_config)

        train_loader = DataLoader(im_dataset,
                                batch_size=train_config['batch_size'],
                                shuffle=True, num_workers=8, pin_memory=True)

    if dataset_config['dataset_name'] == 'imagenet-mini':
        
        train_dir = join(dataset_config["img_folder_path"], 'train')
        val_dir = join(dataset_config["img_folder_path"], 'val')
        
        imagenet_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 5]),
        ])

        # Load datasets
        train_dataset = ImageNetMiniDataset(root=train_dir, transform=imagenet_transform)
        val_dataset = ImageNetMiniDataset(root=val_dir, transform=imagenet_transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    # Instantiate the model
 
    model = DiT(dit_model_config).to(device)
    model.train()

    # This is to save the checkpoints, images, loss plots, etc. specific to a dataset, model config, and training config

    save_tag = f"{dataset_config['dataset_name']} {float(train_config['learning_rate'])} {dit_model_config['embed_dim']} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']}".replace(" ", "_")
   

    # Loading pretrained model if exists

    if os.path.exists(join(args.save_root_path,'models',save_tag + '_dit_model.pth')):
        print('Loaded DiT checkpoint')
        model.load_state_dict(torch.load(join(args.save_root_path,'models', save_tag + '_dit_model.pth'),
                                         map_location=device))


    # Specify training parameters
    num_epochs = train_config['num_epochs']

    ### The AdamW optimizer is used for training

    learning_rate = float(train_config['learning_rate'])
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    ### MSE loss is used for training
    criterion = torch.nn.MSELoss()

    # Run training


    start = time.time()

    acc_steps = 1
    epoch_losses = []
    for epoch_idx in range(num_epochs):
        losses = []
        step_count = 0
        for input in tqdm(train_loader):

            
            step_count += 1


            if dataset_config['dataset_name'] == 'landscapehq':

                # print("Input image shape is ", input.shape)

                img = input.float().to(device)


            if dataset_config['dataset_name'] == 'imagenet-mini':
                
                labels = input[1].to(device)
                img = load_latents_imagenet_mini(input[-1]).float().to(device)


            # Sample random noise
            noise = torch.randn_like(img).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'],
                              (img.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(img, noise, t)

            # Noise prediction from the model using noisy image and timestep
            optimizer.zero_grad()

            if dataset_config['dataset_name'] == 'landscapehq':
                pred = model(noisy_im, t)
            if dataset_config['dataset_name'] == 'imagenet-mini':
                pred = model(noisy_im, t, labels)

            # Loss Computation
            noise = noise.squeeze(1)
            loss = criterion(pred, noise)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1,np.mean(losses)))

        epoch_losses.append(np.mean(losses))

        save_tag = f"{dataset_config['dataset_name']} {learning_rate} {diffusion_config['num_timesteps']} {train_config['num_epochs']} {dit_model_config['num_layers']} {dit_model_config['num_heads']} {dit_model_config['patch_height']}".replace(" ", "_")
        
        if epoch_idx % 2 == 0:  # saving the model every 2 epochs
            torch.save(model.state_dict(), join(args.save_root_path,'models', save_tag + '_dit_model.pth'))

        # Training loss plotting at each epoch
    
        plt.figure(figsize=(10, 10))
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
        plt.title("epoch loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(args.save_root_path,'loss_plots',save_tag + "_epoch_loss.png"))

    print('Done Training ...')
    print('Total time taken: {:.2f} minutes'.format((time.time() - start)/60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DiT training')
    parser.add_argument('--config_file', type=str, default= '/home/sid/DiT/config/landscapehq.yaml', help='Path to the config file')
    parser.add_argument('--save_root_path',default= '/home/sid/DiT/results', type=str)
    # parser.add_argument('--dataset_root_path',default= '/home/rohit/imagenet-mini', type=str)
    args = parser.parse_args()


    train(args)