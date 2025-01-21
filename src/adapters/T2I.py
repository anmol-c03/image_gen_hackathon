from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
import torch

# check the cuda/ mps avialability
if torch.cuda.is_available:
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available:
    device="mps"
else:
    device = "cpu"
# load adapter
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'

euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# load vae model
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                            model_id, vae=vae, 
                            adapter=adapter, 
                            scheduler=euler_a, 
                            torch_dtype=torch.float16, 
                            variant="fp16",
                            ).to(device)

pipe.enable_xformers_memory_efficient_attention()

