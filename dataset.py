# Python Script for DataLoader class


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

import torch
import torchvision

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from diffusers import AutoencoderKL
from torchvision.datasets import ImageFolder
from utils import show_images,load_latents

# from generate_latents_landscape import load_latents


class LandscapesDataset(Dataset):
    def __init__(self,args):
        super().__init__()

        self.args = args

        self.img_folder_path = args["img_folder_path"]

        self.img_resize = args["landscape_img_resize"]

        self.use_latent_features = args["use_latent_features"]

        self.latent_maps_path = args["latent_maps_path"]

        self.images = [join(self.img_folder_path,img_file) for img_file in os.listdir(self.img_folder_path)]


        print("Training mode")
        print(f"Number of images: {len(self.images)}")

        if self.use_latent_features:
            print("Using latents")
            # Load the latent maps
            self.latent_maps = load_latents(self.latent_maps_path)
            #length check
            print(f"Number of latent maps: {len(self.latent_maps)}")
            print("Latent maps loaded")

        else:
            print("Using images")          

            self.img_transform = transforms.Compose([
                    transforms.Resize((self.img_resize,self.img_resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])

    def __len__(self):
        return len(self.latent_maps)

    def __getitem__(self, idx):

        if self.use_latent_features:

            if self.args["dataset_name"] == 'landscape':
                latent = self.latent_maps[self.images[idx]]#.replace('landscape_dataset','DiT/landscape')]
            if self.args["dataset_name"] == 'landscapehq':
                latent = self.latent_maps[idx]

            return latent
        else:
            img = Image.open(self.images[idx]).convert('RGB')

            img = self.img_transform(img)

        # print(img.shape)

            return img


class ImageNetMiniDataset(ImageFolder):
    def __getitem__(self, index):
        # Original tuple: (image, label)
        original_tuple = super().__getitem__(index)
        
        # Get the image file path
        path = self.imgs[index][0]
        
        # Append path to the original tuple
        return original_tuple + (path,)