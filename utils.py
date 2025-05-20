# Python Script for Utility Functions like Visualization functions etc.

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from os.path import join
import torchvision
import glob

import numpy as np


def show_images(images, nrow=None, figsize=(8, 8), title=None,save_path=None):

    # Convert to numpy if tensor
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    if images.ndim == 3:
        # If shape is [C, H, W], convert to [H, W, C]
        if images.shape[0] == 3:
            images = images[None,:]

    
    # If shape is [B, C, H, W], convert to [B, H, W, C]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    # Normalize to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    if images.min() < 0.0:
        images = (images + 1) / 2  # for [-1, 1] inputs

    # Auto grid size if not specified
    N = images.shape[0]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(N)))

    ncol = int(np.ceil(N / nrow))
    
    ############## Creating grid plot #############################

    fig, axes = plt.subplots(nrows=ncol, ncols=nrow, figsize=figsize)


    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis('off')

        if i < N:
            img = images[i].squeeze()
            axes[i].imshow(img)  


    if title:
        fig.suptitle(title)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path,f'{title}.png' if title else 'sample_image'), bbox_inches='tight', dpi=300)
        print(f"Saved image grid to: {save_path}")

    plt.tight_layout()
    plt.show()

def show_image_grid(image_tensor_list,save_path='decoded_images.png'):

    print(image_tensor_list[0].shape)

    if not isinstance(image_tensor_list, torch.Tensor):
        image_tensor = torch.cat(image_tensor_list,dim=0)
    else:
        image_tensor = image_tensor_list

    

    print(image_tensor.shape)

    # Create a grid from the batch of images
    grid = torchvision.utils.make_grid(image_tensor, nrow=int(np.ceil(np.sqrt(image_tensor.shape[0]))), padding=2, normalize=True)

    # Convert the grid to numpy and display
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(save_path, dpi=300)

def generate_latents(args):

    img_folder_path = args.img_folder_path

    images = [join(img_folder_path,img_file) for img_file in os.listdir(img_folder_path)]

    if args.dataset_name == 'landscape' and args.use_latent_features:
        latent_maps = {}
        for i,sample in tqdm(enumerate(landscape_dataloader), total=len(landscape_dataloader), desc="Generating latents"):
            img = sample.to(device)
            print(img.shape)
            latent = vae.encode(img).latent_dist.mean
            latent_maps[images[i]] = latent
            # print(latent.shape)
        print("Latents generated")

    else:
        return None
    #save the latents in .pkl file

    torch.save(latent_maps, os.path.join(os.getcwd(), 'landscape_latent_maps.pkl'))

def load_latents(latent_maps_path=None):


    latents_list = []

    landscape_test_indices = np.load('landscapehq_test_indices.npy', allow_pickle=True)

    for latent_map_file_path in glob.glob(join(latent_maps_path,'*')):

        if int(latent_map_file_path.split('/')[-1].split('_')[0]) not in landscape_test_indices:

            latent_maps = torch.load(latent_map_file_path)

            latents_list.append(latent_maps)

    latent_maps = torch.stack(latents_list)
    
    print("Latents loaded")

   

    return latent_maps


def load_latents_imagenet_mini(img_paths_list):

    latent_maps_path = []

    latents_list = []

    for path in img_paths_list:
        image_file_name = path.split('/')[-1].split('.')[0]
        class_name = path.split('/')[-2]
        latent_map_path = join('/home/rohit/DiT/latent_maps/imagenet-mini',class_name,image_file_name + '_latent.pt')

        latent_maps = torch.load(latent_map_path)

        latents_list.append(latent_maps)

    
   
    # print("Latents loaded")

    return torch.stack(latents_list)