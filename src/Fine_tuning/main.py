#ok
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from typing import Optional

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from peft import LoraConfig
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#ok
@dataclass
class TrainingConfig:
    pretrained_name: str = "stabilityai/stable-diffusion-xl-base-1.0"  # Updated for SDXL
    train_steps: int = 1000
    seed: int = 42
    rank: int = 128
    batch_size: int = 1
    accumulation_steps: int = 4
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    snr_gamma: float = 5.0
    CSV_PATH = '/kaggle/input/carpets/Carpets/desc.csv'
    BASE_IMAGE_DIR = '/kaggle/input/carpets/Carpets'
    output_dir: str = "output"
    resolution: int = 512
    
# ok
def setup_models_for_training(model_name, rank: int = 128):
    # Load models in float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler

    # Freeze all models
    for model in [text_encoder, text_encoder_2, vae, unet]:
        model.requires_grad_(False)
        model.eval()
        
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "processor", "proj_in", "proj_out"
        ],
    )

    unet.add_adapter(unet_lora_config)

    # Ensure LoRA parameters are in float32
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.to(dtype=torch.float32)

    return tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae, scheduler, unet

#ok
class SDXLDataset(Dataset):
    def __init__(self, csv_path, base_image_dir, resolution=1024):
        self.dataframe = pd.read_csv(csv_path)
        self.base_image_dir = base_image_dir
        self.resolution = resolution

        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),  # Added center crop
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.base_image_dir, row['image'])
        prompt = row['prompt']

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            
            # Add checks for NaN or infinity
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError(f"Invalid values in image tensor from {image_path}")
                
            return {
                "pixel_values": image,
                "input_ids": prompt
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a valid alternative or skip this sample
            raise e
        
#ok
def train(
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: TrainingConfig,
    device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    text_encoder.to(device).eval()
    text_encoder_2.to(device).eval()
    vae.to(device).eval()
    unet.to(device).train()

    # Get trainable parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    # Initialize dataset and dataloader
    dataset = SDXLDataset(config.CSV_PATH, config.BASE_IMAGE_DIR, resolution=config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2
    )

    # Training loop
    progress_bar = tqdm(range(config.train_steps))
    global_step = 0
    losses = []
    optimizer.zero_grad()

    for epoch in range(int(math.ceil(config.train_steps / len(dataloader)))):
        for batch in dataloader:
            # Process text embeddings
            with torch.no_grad():
                prompt_embeds_list = []
                
                # First text encoder
                text_inputs = tokenizer(
                    batch["input_ids"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                encoder_output = text_encoder(
                    text_inputs.input_ids,
                    output_hidden_states=True,
                    return_dict=False
                )
                prompt_embeds = encoder_output[-1][-2]
                prompt_embeds_list.append(prompt_embeds)
                
                # Second text encoder
                text_inputs_2 = tokenizer_2(
                    batch["input_ids"],
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                encoder_output_2 = text_encoder_2(
                    text_inputs_2.input_ids,
                    output_hidden_states=True,
                    return_dict=False
                )
                prompt_embeds_2 = encoder_output_2[-1][-2]
                pooled_prompt_embeds = encoder_output_2[0]
                prompt_embeds_list.append(prompt_embeds_2)
                
                # Concatenate embeddings
                prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
                
                # Convert to float32 to prevent NaN
                prompt_embeds = prompt_embeds.to(dtype=torch.float32)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float32)

            # Process images
            latents = vae.encode(
                batch["pixel_values"].to(device, dtype=torch.float32)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Prepare time embeddings
            add_time_ids = torch.cat(
                [
                    torch.tensor([config.resolution, config.resolution]).repeat(latents.shape[0], 1),
                    torch.tensor([0, 0]).repeat(latents.shape[0], 1),
                    torch.tensor([config.resolution, config.resolution]).repeat(latents.shape[0], 1),
                ],
                dim=-1,
            ).to(device, dtype=torch.float32)  # Ensure float32

            try:
                # Make prediction
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids
                    },
                ).sample

                # Compute loss
                if config.snr_gamma > 0:
                    snr = compute_snr(scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Check for NaN loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN or Inf loss detected: {loss.item()}")
                    continue

                # Backward pass
                loss = loss / config.accumulation_steps
                loss.backward()

                # Print shapes for debugging
                print(f"prompt_embeds shape: {prompt_embeds.shape}")
                print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
                print(f"Loss: {loss.item()}")

            except RuntimeError as e:
                print(f"Error in forward pass: {str(e)}")
                print(f"prompt_embeds shape: {prompt_embeds.shape}")
                print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
                continue

            # Gradient clipping and optimization
            if (global_step + 1) % config.accumulation_steps == 0:
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item() * config.accumulation_steps)
            progress_bar.set_postfix({"loss": losses[-1]})
            progress_bar.update(1)

            global_step += 1
            if global_step >= config.train_steps:
                break

        if global_step >= config.train_steps:
            break

    # Save the fine-tuned model
    os.makedirs(config.output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(config.output_dir, "unet_lora"))

    return {"losses": losses}


config = TrainingConfig()
config.lr =  5e-6
config.rank = 62
config.train_steps = 1000
config.snr_gamma = 5.0
config.seed = 42

torch.manual_seed(config.seed)
 # Setup models
models = setup_models_for_training(config.pretrained_name, rank=config.rank)

# Train
outputs = train(*models, config)