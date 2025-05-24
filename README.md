# Implementation of Diffusion Transformer (DiT). Experimentation with DDIM sampler. Experimentation with classifier-free guidance.

Data Preparation:

Datasets considered: [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data), [LandscapesHQ256](https://github.com/universome/alis/blob/master/lhq.md).

Download the datasets from the above links.

LandscapesHQ dataset consists of 90,000 images. It is split into 80,000 train and 10,000 test images. The indices for images in test set is present in 'landscapehq_test_indices.npy' file.

ImageNet-mini comes with its own train and val split. It consists of approx 34k images in train set.


Training: 

1. For training latent_maps for the input images are needed. You can download them from this [link](https://drive.google.com/file/d/168QveHjpkI-TuTU9OMZo3Hxz-4Cpf17f/view?usp=sharing).(Path to the dataset folder should be given in the yaml config files.

2. Put them in the latent_maps folder: /DiT_Assignment/latent_maps/ (this path should be given in the config yaml files.)

```
conda env create -f DL_environment.yml

conda activate DL

cd ~/DiT_Assignment
```
For unconditional training on the  LandscapesHQ dataset.

```
python train.py --config_file ./config/landscapehq.yaml --save_root_path <full_path_to_save_results>
```

For the label conditioned with dropout training for CFG on the ImageNet-mini dataset.

```
python train.py --config_file ./config/imagenet-mini.yaml --save_root_path <full_path_to_save_results>
```

Inference:

To run CFG on the imagenet-mini trained DiT model.

```
cd ~/DiT_Assignment
python sampling_cfg.py --config_file ./config/<config_file_name>.yaml --save_root_path <full_path_to_save_results>
```

To run DDPM sampling on the LandscapesHQ-trained DiT model.

```
cd ~/DiT_Assignment
python sampling.py --config_file ./config/<config_file_name>.yaml --save_root_path <full_path_to_save_results>
```
Evaluation:

FID values are computed based on this repo: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

Use the following command to get the FID between the Generated and Real Images.

```
python -m pytorch_fid <path_to_generated_images_folder> <path_to_real_images_folder> --device cuda:0 
```
Note: This codebase is inspired form the original DiT implementation and [ExplainingAI](https://github.com/explainingai-code/DiT-PyTorch/tree/main)
