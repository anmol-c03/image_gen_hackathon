#!pip install clip_interrogator

import os, subprocess
from clip_interrogator import Config, Interrogator
import csv
import os
from IPython.display import clear_output, display
from PIL import Image
from tqdm import tqdm


def setup():
    install_cmds = [
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', 'clip-interrogator'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))


caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]



config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)


def sanitize_for_filename(prompt: str, max_len: int) -> str:
    name = "".join(c for c in prompt if (c.isalnum() or c in ",._-! "))
    name = name.strip()[:(max_len-4)] # extra space for extension
    return name


def generate_prompt_recursive(folder_path, prompt_mode, output_mode, max_filename_len):
    """
    Generates prompts for images in a nested directory structure and saves to desc.csv or renames files.

    Args:
        folder_path (str): Root directory containing images (and subdirectories).
        prompt_mode (str): Mode for generating prompts.
        output_mode (str): Either 'desc.csv' (save prompts) or 'rename' (rename files).
        max_filename_len (int): Maximum length of filenames when renaming.

    Returns:
        None
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    prompts = []
    file_paths = []  # To store full paths for images

    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                file_paths.append(os.path.join(root, file))

    # Process each image
    for idx, file_path in enumerate(tqdm(file_paths, desc='Generating prompts')):
        if idx > 0 and idx % 100 == 0:
            clear_output(wait=True)

        # Open the image
        image = Image.open(file_path).convert('RGB')

        # Generate prompt
        prompt = image_to_prompt(image, prompt_mode)
        prompts.append(prompt)

        # Display progress
        print(prompt)
        thumb = image.copy()
        thumb.thumbnail([256, 256])
        display(thumb)

        # Handle renaming
        if output_mode == 'rename':
            rel_dir = os.path.relpath(os.path.dirname(file_path), folder_path)
            name = sanitize_for_filename(prompt, max_filename_len)
            ext = os.path.splitext(file_path)[1]
            new_filename = name + ext
            idx = 1
            new_file_path = os.path.join(folder_path, rel_dir, new_filename)

            while os.path.exists(new_file_path):
                print(f'File {new_filename} already exists, trying {idx+1}...')
                new_filename = f"{name}_{idx}{ext}"
                new_file_path = os.path.join(folder_path, rel_dir, new_filename)
                idx += 1

            os.rename(file_path, new_file_path)

    # Save prompts to desc.csv
    if prompts and output_mode == 'desc.csv':
        csv_path = os.path.join(folder_path, 'desc.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['image', 'prompt'])  # Header row
            for file_path, prompt in zip(file_paths, prompts):
                rel_path = os.path.relpath(file_path, folder_path)  # Save relative paths
                writer.writerow([rel_path, prompt])

        print(f"\n\nGenerated {len(prompts)} prompts and saved to {csv_path}, enjoy!")
    elif output_mode == 'rename':
        print(f"\n\nGenerated {len(prompts)} prompts and renamed your files, enjoy!")
    else:
        print(f"Sorry, I couldn't find any images in {folder_path}")

