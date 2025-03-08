from diffusers.utils import load_image, make_image_grid
from controlnet_aux.canny import CannyDetector

# place url or path to image in local machine if available
url="/path/to/your/image"

image = load_image(url)

# Apply the canny detector
# Detect the canny map in low resolution to avoid high-frequency details
image = CannyDetector()(image, detect_resolution=384, image_resolution=1024)#.resize((1024, 1024))





