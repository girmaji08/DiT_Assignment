# Implementation of Diffusion Transformer (DiT). Experimentations with DDIM sampler. Experimentation with classifier-free guidance.

Data Preparation:

Datasets considered: [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data), [LandscapesHQ256](https://github.com/universome/alis/blob/master/lhq.md).

Download the datasets from the above links.


Training: 

File to be executed: train.py
For training latent_maps for the input images are needed. You can download them from this [link]().
Please put them in the latent_maps folder: /repo_name/latent_maps/<dataset_name>/*.pt
Run bash script: run_train.sh. save_root_path and config_file arguments need to be given accordingly.

Inference:

Files to be executed: sampling.py , sampling_cfg.py

Evaluation:

FID values are computed based on this repo: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

Note: This codebase is inspired form the original DiT implementation and [ExplainingAI](https://github.com/explainingai-code/DiT-PyTorch/tree/main)
