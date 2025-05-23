#!/bin/bash

source activate DL

echo "starting sampling.py"
cd ~/DiT_Assignment

./run_with_nohup.sh "python sampling_cfg.py --config_file ./config/imagenet-mini.yaml --save_root_path ./results" "nohup_sampling_run_testing3.out"
