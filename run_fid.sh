#!/bin/bash

# source activate DL

# echo "starting train.py"
# cd ~/DiT

python -m pytorch_fid '/home/sid/DiT/landscape_val' '/home/sid/DiT/results/samples/landscapehq' --device cuda:0