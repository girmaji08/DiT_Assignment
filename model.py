### Python Script for Simple Diffusion Transformer Model

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm
from einops import rearrange, repeat
import os
from os.path import join
import yaml
import math


class PatchEmbedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        
        self.patch_height = args["patch_height"]
        self.patch_width = args["patch_width"]

        self.img_height = args["img_height"]
        self.img_width = args["img_width"]
        self.num_channels = args["latent_channels"]
        self.embed_dim = args["embed_dim"]
        self.batch_size = args["batch_size"]

        self.num_patches = self.img_height // self.patch_height * self.img_width // self.patch_width

        self.patch_dim = self.num_channels * self.patch_height * self.patch_width

        self.patch_embedding = nn.Linear(self.patch_dim, self.embed_dim)


        # self.dropout = nn.Dropout(0.1)


        #### Linear layer initialization  ############
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.constant_(self.patch_embedding.bias, 0)

    def get_2d_sinusoidal_positional_embedding(self,embed_dim, grid_size, device):
        assert embed_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
        grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)

        # grid_h_positions -> (Number of patch tokens,)
        # Flattening the grids
        grid_h_positions = grid[0].reshape(-1)
        grid_w_positions = grid[1].reshape(-1)

        d_model = embed_dim

        #Frequency

        w = torch.exp(- math.log(10000) * torch.arange(start=0, end= d_model // 4, dtype=torch.float32) / (d_model // 4)).to(device)

        # Frequency * Time/Position
        grid_h_emb = grid_h_positions[:, None].repeat(1, d_model // 4) * w
        grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)

        # Concatenate the sine and cosine

        grid_w_emb = grid_w_positions[:, None].repeat(1, d_model // 4) * w
        grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
        pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

        return pos_emb


    def forward(self, x):
        # x: (batch_size, num_channels, img_height, img_width)

        x = rearrange(x, pattern = 'b c (nh hp) (nw wp) -> b (nh nw) (hp wp c)',hp=self.patch_height, wp=self.patch_width)

        x = self.patch_embedding(x)


        ###### 2D sinusoidal position embedding ############

        grid_size_h = self.img_height // self.patch_height
        grid_size_w = self.img_width // self.patch_width



        pos_embed = self.get_2d_sinusoidal_positional_embedding(embed_dim=self.embed_dim,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                 device=x.device)


        # print(f"Positional embedding shape: {pos_embed.shape}")
        ######## Adding position embedding to the patch embedding ##########

        x += pos_embed


        # x = self.dropout(x)


        return x

class TimeEmbedding(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.embed_dim = args["embed_dim"]  # timestep embed dim same as the model embed dim
        self.batch_size = args["batch_size"]  # batch size for the model

        self.time_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.normal_(self.time_proj[2].weight, std=0.02)


    def get_1d_sinusoidal_positional_embedding(self,t, embed_dim):

        d_model = embed_dim
        
        #Frequency
        w = torch.exp(- math.log(10000) * torch.arange(start=0, end= d_model // 2, dtype=torch.float32) / (d_model // 2)).to(t.device)

        # Frequency * time/position
        w_t = t[:, None].repeat(1, embed_dim // 2) * w


        t_embed = torch.cat([torch.sin(w_t),torch.cos(w_t)],dim=-1)  
        return t_embed

    def forward(self, t):

        t_embed = self.get_1d_sinusoidal_positional_embedding(t, self.embed_dim)

        t_embed = self.time_proj(t_embed)

        return t_embed

class LabelEmbedding(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.num_classes = args['num_classes']
        self.cfg_drop_prob = args['cfg_drop_prob']
        self.use_cfg = args['use_cfg']

        self.embed_dim = args['embed_dim']

        if self.cfg_drop_prob > 0:
            self.label_embeddings = nn.Embedding(self.num_classes + 1,self.embed_dim)  # Adding NULL embedding as the last one
        else:
            self.label_embeddings = nn.Embedding(self.num_classes,self.embed_dim)
    
    def forward(self,labels):

        if self.training and self.cfg_drop_prob > 0 and self.use_cfg:  # CFG logic implementation of dropping labels randomly
            
            # print("Labels are being dropped during training")

            drop_labels_bool = torch.randn(labels.shape[0]).to(labels.device) < self.cfg_drop_prob

            final_labels = torch.where(drop_labels_bool,self.num_classes,labels)

            label_embeddings = self.label_embeddings(final_labels)

            return label_embeddings

        else:      # The case of full conditional training and inference
            
            # print(" Labels are not dropped during inference")

            return self.label_embeddings(labels)


class MultiHeadAttention(nn.Module):
    def __init__(self,args):

        super().__init__()

        self.num_heads = args["num_heads"]

        self.embed_dim = args["embed_dim"]

        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = nn.Linear(self.embed_dim,3 * self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # self.attn_drop = nn.Dropout(0.1)

    def forward(self,x):


        q,k,v = self.qkv_proj(x).chunk(3,dim = -1)


        q = rearrange(q,pattern='b np (nh h_dim) -> b nh np h_dim', nh= self.num_heads,h_dim = self.head_dim)


        k = rearrange(k,pattern='b np (nh h_dim) -> b nh np h_dim', nh= self.num_heads,h_dim = self.head_dim)


        v = rearrange(v,pattern='b np (nh h_dim) -> b nh np h_dim', nh= self.num_heads,h_dim = self.head_dim)

        attn_weights = torch.matmul(q,k.transpose(-2,-1))*(self.head_dim**(-0.5))

        attn_weights = F.softmax(attn_weights,dim = -1)

        # attn_weights = self.attn_drop(attn_weights)

        output = torch.matmul(attn_weights,v)

        output = rearrange(output, pattern='b nh np h_dim -> b np (nh h_dim)')

        output = self.out_proj(output)

        return output # shape: B N embed_dim

class TransformerBlock(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.embed_dim = args["embed_dim"]


        ff_hidden_dim = 4 * self.embed_dim

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1E-6)

        self.attn_block = MultiHeadAttention(args)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1E-6)

        # MLP layer 
        self.mlp_block = nn.Sequential(
            nn.Linear(self.embed_dim, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.embed_dim),
        )

        # Adaptive norm layer used to generate 6 scale and shift parameters.

        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)


    def forward(self, x, condition): #time or class conditioning
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)

        (pre_attn_shift, pre_attn_scale, post_attn_scale,
         pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params  # pre and post scale and shift.Post scale and shift are used for MLP and Attention outputs

        out = x

        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1))
                            + pre_attn_shift.unsqueeze(1))    # pre scale and shift are for LayerNorm

        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)

        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
                           pre_mlp_shift.unsqueeze(1))

        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)

        return out

class DiT(nn.Module):
    def __init__(self, args):
        super().__init__()

        print(args)

        self.num_layers = args['num_layers']
        self.image_height = args["img_height"]
        self.image_width = args["img_width"]
        self.im_channels = args["latent_channels"]
        self.embed_dim = args['embed_dim']
        self.patch_height = args['patch_height']
        self.patch_width = args['patch_width']
        self.use_cond = args['use_cond']

        self.num_classes = args['num_classes']
        self.cfg_drop_prob = args['cfg_drop_prob']
        self.use_cfg = args['use_cfg']

        # Number of patches along height and width
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(args)

        self.time_embed_layer = TimeEmbedding(args)

        if self.use_cond:    # This is for conditioning on the labels. We use labels for the conditioning case 
            self.label_embed_layer = LabelEmbedding(args)



        # All Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(self.num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the output from DiT blocks
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.embed_dim,
                                  self.patch_height * self.patch_width * self.im_channels)

        ############################
        # DiT Layer Initialization #
        ############################

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, y=None):
        # Patchify
        x = x.squeeze(1)  
        out = self.patch_embed_layer(x)

        t_emb = self.time_embed_layer(t) # Time embedding

        # print(y,self.use_cond)

        if y is not None and self.use_cond:
            
            y_emb = self.label_embed_layer(y)  # This is for conditioning on the labels
            condition = t_emb + y_emb

        else:
            condition = t_emb  # This is for unconditional generation

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, condition)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)   # Final Linear layer
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)


        return out

    def forward_with_cfg(self,x,t,y,cfg_weight = 0.5):  # This logic is taken from the original DiT implementation.
        
        x_noise_1 = x[: len(x) // 2]  # The inputs x here is a duplication of noise samples, y is combined class labels and null labels of equal size

        x_noise_duplicated = torch.cat([x_noise_1, x_noise_1], dim=0)

        noise_pred = self.forward(x_noise_duplicated, t, y)

        # print("Doing CFG")

        cond_noise_pred, uncond_noise_pred = torch.split(noise_pred, len(noise_pred) // 2, dim=0)

        final_noise_pred = uncond_noise_pred + cfg_weight * (cond_noise_pred - uncond_noise_pred) # The main logic of combining unconditional and conditional predictions using guidance weight

        final_noise_pred = torch.cat([final_noise_pred, final_noise_pred], dim=0)

        return final_noise_pred, cond_noise_pred, uncond_noise_pred