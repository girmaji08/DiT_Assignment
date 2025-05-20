import glob
import os
from os.path import join
import cv2

import numpy as np
from PIL import Image

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import datasets, transforms
import torch
import torchvision

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from utils import show_images,show_image_grid
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description='Diffusion models assignment')
parser.add_argument('--dataset_name',type=str,default='landscape',help='Name of the dataset')
parser.add_argument('--img_folder_path',type=str,default='/home/sid/landscape_dataset')

parser.add_argument('--landscape_img_resize',type=int,default= 256)
parser.add_argument('--use_latent_features',type=bool,default= True)
parser.add_argument('--latent_maps_path',type=str,default='')
parser.add_argument('--visualize_latents',type=bool,default=False)


args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device 

class LandscapesDataset(Dataset):
    def __init__(self,args):
        super().__init__()

        self.img_folder_path = args.img_folder_path

        self.img_resize = args.landscape_img_resize

        self.images = [join(self.img_folder_path,img_file) for img_file in os.listdir(self.img_folder_path)]

        self.img_transform = transforms.Compose([
                transforms.Resize((self.img_resize,self.img_resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # if self.use_latents:
        #     latent = self.latent_maps[self.images[index]]
        #     return latent

        # else:
        img = Image.open(self.images[idx]).convert('RGB')

        img = self.img_transform(img)

        # print(img.shape)

        return img

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        # Original tuple: (image, label)
        original_tuple = super().__getitem__(index)
        
        # Get the image file path
        path = self.imgs[index][0]
        
        # Append path to the original tuple
        return original_tuple + (path,)


def generate_latents(args,dataloader):

    img_folder_path = args.img_folder_path
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)

    # Construct the output path
    output_dir = os.path.join(os.getcwd(), 'latent_maps')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    for i, (sample,ids,img_file_path) in tqdm(enumerate(dataloader)):

        img = sample.to(device)

        latent = vae.encode(img).latent_dist.mean

        base_name = img_file_path[0].split('/')[-2]


        filename = img_file_path[0].split('/')[-1].split('.')[0]

        os.makedirs(os.path.join(output_dir,args.dataset_name,base_name),exist_ok=True)

        latent_path = os.path.join(output_dir,args.dataset_name,base_name, f"{filename}_latent.pt")

        torch.save(latent.cpu(), latent_path)  # Save to file

        

    print("Latents generated")


def load_latents(latent_maps_path=None):


    latents_list = []

    for latent_map_file_path in glob.glob(join(latent_maps_path,'*','*')):

        latent_maps = torch.load(latent_map_file_path)

        latents_list.append(latent_maps)

    torch.stack(latents_list)
    
    print("Latents loaded")

   

    return latent_maps

def load_latents_imagenet_mini(latent_maps_path=None):

    latent_maps_path = []

    for path in latent_maps_path:
        image_file_name = path.split('/')[-1].split('.')[0]
        class_name = image_file_name.split('_')[0]
        latent_map_path = join('/home/rohit/DiT/latent_maps/',class_name,image_file_name + '_latent.pt')
        latent_maps_path.append(latent_map_path)


    latents_list = []

    for latent_map_file_path in latent_maps_path:

        latent_maps = torch.load(latent_map_file_path)

        latents_list.append(latent_maps)

    torch.stack(latents_list)
    
    print("Latents loaded")

   

    return latent_maps

def decode_latents(latent_maps):
    # Decode the latents
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)

    decoded_images = {}

    images_list = []

   

    for i in range(latent_maps.shape[0]):

        latent = latent_maps[i].to(device).unsqueeze(0)

        final_image = (vae.decode(latent).sample)

        final_image = final_image.clamp(-1,1)
        final_image = (final_image+1)/2

        images_list.append(final_image)
    
        if i == 9:
            break

    show_image_grid(images_list)

    # show_images(final_image, figsize=(10, 10), title="Decoded Images from Latents",save_path = '/home/sid/DiT/decoded_images.png')

    




if args.dataset_name == 'landscape':


    landscape_dataset = LandscapesDataset(args)

    # pytorch dataloader
    landscape_dataloader = torch.utils.data.DataLoader(landscape_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    if os.path.exists(args.latent_maps_path):
        print("Latent maps already exist")
    
    else:
        generate_latents(args,landscape_dataloader)

    if args.visualize_latents:
        # Visualize the 9 latents maps

        # load the latent maps
        print("visualizing latent maps")

        latent_maps = load_latents(latent_maps_path = args.latent_maps_path)

        decode_latents(latent_maps)

if args.dataset_name == 'landscapehq':


    landscape_dataset = LandscapesDataset(args)

    # pytorch dataloader
    landscape_dataloader = torch.utils.data.DataLoader(landscape_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    generate_latents(args,landscape_dataloader)

    if args.visualize_latents:
        # Visualize the 9 latents maps

        # load the latent maps
        print("visualizing latent maps")
        
        latent_maps = load_latents(latent_maps_path = args.latent_maps_path)

        decode_latents(latent_maps)

if args.dataset_name == 'imagenet-mini':

    imagenet_root = "/home/rohit/imagenet-mini"
    train_dir = f"{imagenet_root}/train"
    val_dir = f"{imagenet_root}/val"

    # Define transforms (standard ImageNet preprocessing)
    imagenet_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 5]),
    ])

    # Load datasets
    train_dataset = ImageFolderWithPaths(root=train_dir, transform=imagenet_transform)
    val_dataset = ImageFolderWithPaths(root=val_dir, transform=imagenet_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


    # if os.path.exists(args.latent_maps_path) and len(os.listdir(args.latent_maps_path)) != 0:
    #     print("Latent maps already exist")
    
    # else:
    generate_latents(args,train_loader)

    if args.visualize_latents:
        # Visualize the 9 latents maps

        # load the latent maps
        print("visualizing latent maps")

        latent_maps = load_latents(latent_maps_path = args.latent_maps_path)

        decode_latents(latent_maps)






































            # latent_maps = {}
    # for i,sample in tqdm(enumerate(dataloader)):
    #     img = sample.to(device)
    #     # print(img.shape)
    #     latent = vae.encode(img).latent_dist.mean#.mul_(0.18215)#vae.encode(img).latent_dist.mean
    #     latent_maps[images[i]] = latent
    #     # print(latent.shape)
    # print("Latents generated")

    # #save the latents in .pkl file

    # torch.save(latent_maps, os.path.join(os.getcwd(), 'landscape_latent_maps_normalized.pkl'))