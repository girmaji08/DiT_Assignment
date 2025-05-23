from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import os
from torchvision.utils import save_image

save_path = '~/DiT_Assignment/intermediates_DDIM'

# Load scheduler and pipeline
scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Folder to save intermediate images
os.makedirs(save_path, exist_ok=True)

# Callback to save intermediates
def save_intermediate(step: int, timestep: int, latents: torch.FloatTensor):
   
    with torch.no_grad():
        image = pipe.vae.decode(latents / 0.18215).sample  # Latent scaling factor for Stable Diffusion
        save_image((image.clamp(-1, 1) + 1) / 2, f"~/DiT_Assignment/intermediates_DDIM/SD_{step:03d}.png")


pipe(prompt="A futuristic city at sunset", num_inference_steps=200, callback=save_intermediate, callback_steps=2)