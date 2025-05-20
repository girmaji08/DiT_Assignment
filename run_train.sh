#!/bin/bash

source activate DL

echo "starting train.py"
cd ~/DiT

python /home/sid/DiT/train.py --config_file /home/sid/DiT/config/landscapehq.yaml