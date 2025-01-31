'''
This is the .py file for image generation pipeline given a reference image 
This is the .py for corresponding infer.ipynb , for proper visualization use infer.ipynb
'''


from PIL import Image
from CLIP import *
from clip_interrogator import Config, Interrogator
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline



folder_path = "/path/to/your/images"  # desc.csv is saved in this folder_path, so make give read and write access
prompt_mode = 'best' 
output_mode = 'desc.csv' 
max_filename_len = 128 


# model used for caption generation
caption_model_name = 'blip-large' 
clip_model_name = 'ViT-L-14/openai' 

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)
ci.config.quiet = True

# generate prompt for stable diffusion using clip interrogator
generate_prompt(folder_path,prompt_mode,output_mode,max_filename_len)

# Load the CSV file
csv_path = "/path/to/your/csv"
df = pd.read_csv(csv_path)

#load diffusion-2-1 base model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("/content/pytorch_lora_weights.safetensors")

pipe = pipe.to("cuda")

negative_prompt = "blurry, low quality, distorted, overexposed, watermark"

for i in range(df.shape[0]):
    image=df['image'][i]
    prompt=df['prompt'][i]
    full_image_path=os.path.join(folder_path,image)
    init_image = Image.open(full_image_path).convert("RGB")  # Provide path to your image
    init_image = init_image.resize((512, 512)) 

    strength = 0.7  # How much the new image differs from the original (0 = no change, 1 = fully new)
    guidance_scale = [4,4.5,5,5.5,6]  # Controls how strongly the model follows the text prompt

    for k in range(len(guidance_scale)):
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale[k]
        ).images[0]
        os.makedirs(image,exist_ok=True)
        output_image.save(f"{image}/output_image{k}.png")





