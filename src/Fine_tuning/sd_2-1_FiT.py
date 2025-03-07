'''
This py script can be used to Fine Tune stable diffusion version 2.1.
It consists of an experimental approach to preserve pattern and geometry/structural features in input 
images by Fine Tuning in latent and pixel space combined with a hyperparameter lambda.
'''


from pathlib import Path
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import login
from peft import LoraConfig
import torch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from functools import partial
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from diffusers.training_utils import compute_snr
import torch.nn as nn 


def get_models(model_name, dtype=torch.float16):
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(dtype=dtype)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(dtype=dtype)
    return tokenizer, text_encoder, vae, scheduler, unet

def setup_models_for_training(model_name, rank: int=128):
    tokenizer, text_encoder, vae, scheduler, unet = get_models(model_name)

    # freeze all weights
    for m in (unet, text_encoder, vae):
        for p in m.parameters():
            p.requires_grad = False

    # config LoRA
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    # set trainaible weights to float32
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.to(dtype=torch.float32)

    return tokenizer, text_encoder, vae, scheduler, unet

def get_lora_params(unet):
    return [p for p in filter(lambda p: p.requires_grad, [p for p in unet.parameters()])]

from dataclasses import dataclass
@dataclass
class TrainingConfig():
    train_steps: int = 100
    lr: float = 1e-5
    batch_size: int = 4
    accumulation_steps: int = 2
    rank: int = 128
    max_grad_norm: float = 1.0
    pretrained_name: str = "stabilityai/stable-diffusion-2-1-base"
    snr_gamma: float = -1
    seed: int = -1
    CSV_PATH = '/mnt/Enterprise3/aavash/tmp/exp/Carpet_final/Carpets/final_merged.csv'

    BASE_IMAGE_DIR = '/mnt/Enterprise3/aavash/tmp/exp/Carpet_final/Carpets'



import os
class CarpetWallpaperDataset(Dataset):
    def __init__(self, csv_path, base_image_dir,tokenizer):
        self.dataframe = pd.read_csv(csv_path)
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.train_tranforms = transforms.Compose(
              [
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Resize((512,512)),
                  transforms.Normalize([0.5], [0.5]),
              ]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Construct full image path
        relative_path = self.dataframe.iloc[idx]['image']
        full_image_path = os.path.join(self.base_image_dir, relative_path)

        # Load image
        image = Image.open(full_image_path).convert('RGB')

        image_tensor=self.train_tranforms(image)


        input_ids = self.tokenizer(
            self.dataframe.iloc[idx]['prompt'],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"][0]
        # Get corresponding text prompt
        # text_prompt = self.dataframe.iloc[idx]['prompt']

        return {"pixel_values": image_tensor,
                 "input_ids": input_ids}

class loss_fn(nn.Module):
    def __init__(self,original,generated):
        super().__init__()
        self.original = original
        self.generated = generated
        self.model_vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.model_vgg = self.model_vgg.half()
        self.model_vgg.to('cuda')
        for params in self.model_vgg.parameters():
          params.requires_grad=False
        self.model_vgg.eval()
        self.feature_layers = ['3', '15', '25']  # to extract low, mid and high level features using VGG16


    def forward(self):
        orig_features = self.extract_features(self.original)
        gen_features = self.extract_features(self.generated)
        pattern_loss= self.pattern_loss(orig_features,gen_features)
        return pattern_loss



    def extract_features(self,x):
        features = []
        x=x.to('cuda')
        feature = x 

        self.model_vgg.eval()
        for name, module in self.model_vgg.features._modules.items():
            feature = module(feature)
            if name in self.feature_layers:
                features.append(feature)
        return features

    def pattern_loss(self,orig_features,gen_features):
        loss_p=0
        for o,g in zip(orig_features,gen_features):
            loss_p+=F.mse_loss(o,g)
        return loss_p

def train(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: TrainingConfig,
    device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_params = get_lora_params(unet)

    text_encoder.to(device).eval()
    vae.to(device).eval()
    unet.to(device).train()

    # data set
    train_dataset = CarpetWallpaperDataset(config.CSV_PATH,config.BASE_IMAGE_DIR, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # optimizer
    steps_per_epoch = math.ceil(len(train_dataloader) / config.accumulation_steps)
    epochs = math.ceil(config.train_steps / steps_per_epoch)

    lr = config.lr * config.accumulation_steps * config.batch_size
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    # progress bar setup
    global_step = 0
    progress_bar = tqdm(
        range(config.train_steps),
        desc="Steps"
    )

    print(f"configs: {config}")
    print(f"epochs: {epochs}")
    print(f"steps per epoch: {steps_per_epoch}")
    print(f"total steps: {config.train_steps}")
    print(f"accumulation steps: {config.accumulation_steps}")
    print(f"total batch size: {config.batch_size * config.accumulation_steps}")
    print(f"lr: {lr}")

    losses = []
    for _ in range(epochs):
        for step, batch in enumerate(train_dataloader):
            bs = batch["input_ids"].shape[0]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,)).long().to(device)

                with torch.no_grad():
                    batch["pixel_values"] = batch["pixel_values"].type(torch.float16)
                    latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                with torch.no_grad():
                    pred_images = vae.decode(latents / vae.config.scaling_factor).sample




                pred_normalized = (pred_images / 2 + 0.5).clamp(0, 1)
                target_normalized = ( batch["pixel_values"] / 2 + 0.5).clamp(0, 1)

                pixel_loss=loss_fn(pred_normalized,target_normalized)()
                # pixel_loss=F.mse_loss(pred_normalized, target_normalized)

                if config.snr_gamma > 0:
                    # should converge faster with snr_gamma, however works well with unweighted mse
                    # https://arxiv.org/abs/2303.09556
                    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
                    snr = compute_snr(scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    mse_loss_weights = mse_loss_weights / snr
                    loss = F.mse_loss(noise_pred, noise, reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    latent_loss = loss.mean()
                else:
                    latent_loss = F.mse_loss(noise_pred, noise, reduction="mean")

            global_step+=1

            total_loss=latent_loss+0.24*pixel_loss

            scaler.scale(total_loss).backward()

            if global_step % config.accumulation_steps == 0:
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                progress_bar.update(1)

            losses.append(total_loss.item())

            progress_bar.set_postfix({"loss": losses[-1]})
            if global_step / config.accumulation_steps >= config.train_steps:
                break

    return {
        "losses": losses
    }


config = TrainingConfig()
config.lr = 1e-5
config.rank = 62
config.train_steps = 1000
config.snr_gamma = 5.0
config.seed = 42

torch.manual_seed(config.seed)

models = setup_models_for_training(config.pretrained_name, rank=config.rank)

outputs = train(
    *models,
    config,
)
with open("loss.txt", "w") as f:
    f.write(str(outputs['losses']))
    f.write("\n")


unet = models[-1]
unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
StableDiffusionPipeline.save_lora_weights(
    save_directory="./out_2_1",
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=True,
)