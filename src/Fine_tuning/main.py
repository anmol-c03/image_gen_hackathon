import torch
import torch.cuda as cuda
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import cv2
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from peft import LoraConfig, get_peft_model
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F


device = torch.device("cuda" if cuda.is_available() else "cpu")


class CustomImageVariationLoss(torch.nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def extract_edges(self, image):
        if isinstance(image, torch.tensor):
            image = image.permute(1, 2, 0).numpy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        return torch.from_numpy(edges).float()/255.0

    def text_image_alignment_loss(self, generated_image, text_prompt):
        # Compute CLIP text-image similarity
        inputs = self.clip_processor(text=text_prompt, images=generated_image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        return -outputs.logits_per_image.mean()

    def structural_preservation_loss(self, original_image, generated_image):
        # Compute structural similarity using edge detection and feature matching
        original_edges = self.extract_edges(original_image)
        generated_edges = self.extract_edges(generated_image)

        # Structural preservation metric
        structural_loss = F.mse_loss(original_edges, generated_edges)
        return structural_loss

    def forward(self, original_image, generated_image, text_prompt):
        text_alignment = self.text_image_alignment_loss(generated_image, text_prompt)
        structural_preserve = self.structural_preservation_loss(original_image, generated_image)

        # Weighted combination of losses
        total_loss = 0.6 * text_alignment + 0.4 * structural_preserve
        return total_loss
    

def setup_lora_model(base_model_path):
    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank adaptation
        lora_alpha=32,  # Scaling factor
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none"
    )

    # Load base Stable Diffusion model
    model = StableDiffusionPipeline.from_pretrained(base_model_path)

    unet = model.unet

    lora_model = get_peft_model(unet, lora_config)

    model.unet = lora_model

    return model

class CarpetWallpaperDataset(Dataset):
    def __init__(self, csv_path, base_image_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        print(self.dataframe.head)
        self.base_image_dir = base_image_dir
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Construct full image path
        relative_path = self.dataframe.iloc[idx]['image']
        full_image_path = os.path.join(self.base_image_dir, relative_path)

        # Load image
        image = Image.open(full_image_path).convert('RGB')

        image = self.transform(image)

        # Get corresponding text prompt
        text_prompt = self.dataframe.iloc[idx]['prompt']

        return {
            'image': image,
            'text_prompt': text_prompt
        }
    

def prepare_dataloader(csv_path, base_image_dir, batch_size=4):
    dataset = CarpetWallpaperDataset(csv_path, base_image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return dataloader


# Modified training function to work with the new dataset
def train_image_variation_model(
    csv_path,
    base_image_dir,
    custom_loss_fn,
    num_epochs=5,
    learning_rate=1e-4
):
    # Setup model and optimizer
    lora_model = setup_lora_model("stabilityai/stable-diffusion-xl-base-1.0")

    pipeline = lora_model
    pipeline.to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.AdamW(lora_model.unet.parameters(), lr=learning_rate)

    # Prepare DataLoader
    dataloader = prepare_dataloader(csv_path, base_image_dir)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch['image'].to(device)
            prompts = batch['text_prompt']
            print(images[0].shape, len(prompts))
            added_cond_kwargs = {"text_embeds": prompts} if prompts is not None else {}
            # Image-to-image generation using mixed precision training
            with torch.cuda.amp.autocast():
                generated_images = pipeline(
                    prompt=prompts,
                    image=images,
                    strength=0.75,  # Controls image variation intensity
                    guidance_scale=7.5,
                    added_cond_kwargs={"text_embeds": prompts}  # Ensure added_cond_kwargs is not None
                ).images
            print(images[0].shape, generated_images[0].shape)
            # Compute custom loss for each image in batch
            total_loss = 0
            for orig_img, gen_img, prompt in zip(images, generated_images, prompts):
                batch_loss = custom_loss_fn(
                    original_image=orig_img,
                    generated_image=gen_img,
                    text_prompt=prompt
                )
                total_loss += batch_loss

            # Backpropagate and update model
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch+1}/{num_epochs} completed")

    return lora_model



# Usage example
# if __name__ == "__main__":
    # Paths to configure
CSV_PATH = '/mnt/Enterprise2/aavash/cpt/image_gen_hackathon/images/Carpets/desc.csv'
BASE_IMAGE_DIR = '/mnt/Enterprise2/aavash/cpt/image_gen_hackathon/images/Carpets'
print(1)
# Initialize custom loss function
custom_loss_fn = CustomImageVariationLoss()
print(2)
# Train the model
trained_model = train_image_variation_model(
    csv_path=CSV_PATH,
    base_image_dir=BASE_IMAGE_DIR,
    custom_loss_fn=custom_loss_fn
)
