# Implementation of Diffusion Transformer (DiT). Implementation of DDIM sampler. Experimentation with classifier-free guidance.

## **Data Preparation:**

- Datasets considered: [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data), [LandscapesHQ256](https://github.com/universome/alis/blob/master/lhq.md).

- The LandscapesHQ dataset consists of 90,000 images. It is split into 80,000 training and 10,000 test images. The indices for images in the test set are present in ***landscapehq_test_indices.npy*** file.

- The ImageNet-mini comes with its own train and val split. It consists of approximately 34k images in the training set.

Follow these steps to prepare the input data for training.

1. Download the datasets from the above links.

2. Latent maps are required for training with the input images. You can download them from this [link](https://drive.google.com/file/d/168QveHjpkI-TuTU9OMZo3Hxz-4Cpf17f/view?usp=sharing). 

3. After downloading, place the latent maps in the following directory: /DiT_DDIM_CFG/latent_maps/

Make sure to specify the path to the dataset folder in the corresponding YAML configuration files. Ensure the latent maps path is correctly set in the YAML config files as well.

## **Training:**

Set up the Conda virtual environment using the following steps:
```
conda env create -f DL_environment.yml

conda activate DL

```
For unconditional training on the  LandscapesHQ dataset.

```
cd ~/DiT_DDIM_CFG
python train.py --config_file ./config/landscapehq.yaml --save_root_path <full_path_to_save_results>
```

For the label conditioned with dropout training for CFG on the ImageNet-mini dataset.

```
cd ~/DiT_DDIM_CFG
python train.py --config_file ./config/imagenet-mini.yaml --save_root_path <full_path_to_save_results>
```

## **Inference:**

To run CFG on the imagenet-mini trained DiT model.

```
cd ~/DiT_DDIM_CFG
python sampling_cfg.py --config_file ./config/<config_file_name>.yaml --save_root_path <full_path_to_save_results>
```

To run DDPM sampling on the LandscapesHQ-trained DiT model.

```
cd ~/DiT_DDIM_CFG
python sampling.py --config_file ./config/<config_file_name>.yaml --save_root_path <full_path_to_save_results>
```

To run DDIM Sampler on the LandscapesHQ-trained DiT model.

```
cd ~/DiT_DDIM_CFG
python sampling_ddim.py --config_file ./config/<config_file_name>.yaml --save_root_path <full_path_to_save_results>
```

## **Evaluation:**

FID values are computed based on this repo: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

Use the following command to get the FID between the Generated and Real Images.

```
python -m pytorch_fid <path_to_generated_images_folder> <path_to_real_images_folder> --device cuda:0 
```
Note: This codebase is inspired form the original DiT implementation and [ExplainingAI](https://github.com/explainingai-code/DiT-PyTorch/tree/main)
