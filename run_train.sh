#!/bin/bash

source activate DL

echo "starting train.py"
cd ~/DiT_Assignment

# python train.py --config_file ./config/landscapehq.yaml --save_root_path ~/DiT_Assignment/results

CUDA_VISIBLE_DEVICES=1, python train.py --config_file ./config/imagenet-mini.yaml --save_root_path ~/DiT_Assignment/results