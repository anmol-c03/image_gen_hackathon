'''
If one have image->prompt dataset, then can use this pipeline for image generation.
It employs multi experts architecture and activates either T2I or ip adpater based on the image content.
To use this pipeline, we have generated a dataset with image, prompt, rationale from VLMS and their rationale
to categorize whether T2I or ip adpater shoudl be used for generation.
The dataset foramt looks like

image_path,prompt,rationale,Recommended Pipeline
'''

import torch
import pandas as pd
import os
from PIL import Image
from diffusers import (
    AutoPipelineForImage2Image, 
    AutoPipelineForText2Image,
    StableDiffusionXLAdapterPipeline, 
    T2IAdapter, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from controlnet_aux.canny import CannyDetector

# Helper function to load and resize image
def load_image(image_path, base_path):
    absolute_path = os.path.join(base_path, image_path)
    try:
        image = Image.open(absolute_path)
    except Exception as e:
        if image_path.endswith('.png'):
            fallback_path = absolute_path.rsplit('.', 1)[0] + '.jpg'
            image = Image.open(fallback_path)
        else:
            raise e
    return image

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Initialize pipelines only when needed
pipelines = {}

def get_ip_adapter_pipeline():
    if "ip_adapter" not in pipelines:
        print("Loading IP-Adapter pipeline...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch_dtype
        )
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.8)
        pipe.enable_model_cpu_offload()
        pipelines["ip_adapter"] = pipe
    return pipelines["ip_adapter"]

def get_t2i_pipeline():
    if "t2i" not in pipelines:
        print("Loading T2I-Adapter pipeline...")
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-canny-sdxl-1.0", 
            torch_dtype=torch_dtype, 
            varient="fp16"
        )
        
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
        
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id, 
            vae=vae, 
            adapter=adapter, 
            scheduler=euler_a, 
            torch_dtype=torch_dtype,
            variant="fp16",
        )
        pipe.enable_model_cpu_offload()
        pipelines["t2i"] = pipe
        pipelines["canny_detector"] = CannyDetector()
    return pipelines["t2i"], pipelines["canny_detector"]


def process_ip_adapter(row, image, image_path, subscript):
    pipe = get_ip_adapter_pipeline()
    image_resized = image.resize((1280, 1280))
    
    for k in range(1):
        prompt = row[f'prompt_{k}'].strip('"')
        gen_image = pipe(
            prompt=prompt,
            ip_adapter_image=image_resized,
            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            num_inference_steps=30,
            height=1280,
            width=960,
        ).images[0]
        
        gen_image = gen_image.resize(image.size, Image.LANCZOS)
        output_path = f'output_{image_path.split(".")[0]}{subscript[k]}.png'
        gen_image.save(output_path)
        print(f"Saved {output_path}")

def process_t2i(row, image, image_path, subscript):
    pipe, canny_detector = get_t2i_pipeline()
    canny_image = canny_detector(image, detect_resolution=500, image_resolution=1024)
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality"
    
    for k in range(1):
        prompt = row[f'prompt_{k}'].strip('"')
        gen_image = pipe(
            prompt=f'intricate details, Beautiful, 4k quality, vibrant colors, bright, carpet, {prompt}',
            negative_prompt=negative_prompt,
            image=canny_image.resize(image.size),
            num_inference_steps=30,
            guidance_scale=6,
            adapter_conditioning_scale=0.9,
            adapter_conditioning_factor=1
        ).images[0]
        
        gen_image = gen_image.resize(image.size, Image.LANCZOS)
        output_path = f'output_{image_path.split(".")[0]}{subscript[k]}.png'
        gen_image.save(output_path)
        print(f"Saved {output_path}")

def main():
    # Configuration
    csv_path = "/path/to/provided/csv"  # Update with your actual path
    images_path = "/path/to/the/test/set/images"  # Update with your actual path
    subscript = ["a", "b", "c", "d", "e"]
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    df["Recommended Pipeline"] = df["Recommended Pipeline"].str.lower()
    
    # Process each row
    for i, row in df.iterrows():
        pipeline_type = row["Recommended Pipeline"]
        image_path = row["image_path"]
        
        print(f"Processing {i+1}/{len(df)}: {image_path} with {pipeline_type}")
        
        try:
            # Load the image
            image = load_image(image_path, images_path)
            
            # Process based on pipeline type
            if pipeline_type == "ip adapter" or pipeline_type == "ip_adapter":
                process_ip_adapter(row, image, image_path, subscript)
            elif pipeline_type == "t2i":
                process_t2i(row, image, image_path, subscript)
            else:
                print(f"Unknown pipeline type: {pipeline_type}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

if __name__ == "__main__":
    main()