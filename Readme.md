# Image Generation 
This project explores the task of generating multiple high-quality variations of a given input image of carpets, wallpapers or rugs using Stable Diffusion. The goal is to produce five distinct images that retain the abstract nature and patterns of the input while introducing variations in color, shape, and layout. We investigated various techniques, including ControlNet, T2I (Text-to-Image) Adapter and IP Adapter, each with its own strengths and limitations. While the T2I Adapter excelled in preserving intricate patterns, it struggled to introduce significant variations. Conversely, the IP Adapter facilitated neural style transfer but was insufficient for generating meaningful diversity on its own. So rather than using a single adapter, we propose using both based on complexity and abstractness in an image. We devised a LLM-enhanced pipeline to intelligently identify the complexity of an image and then used appropriate pipeline based on that. Furthermore, for domain adaptation, we recommend LoRA Fine Tuning which smoothens out the further generation.

We sucessfully generated 5 different images for 100 images ( total of 500 images cpu offloadings without significant decrease in performance).

## System Pipeline
1. Fine Tuning Pipeline:
![FiT Pipeline](https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/Latent_pixel_space_FiT.png)

<!-- <p align="center">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/Latent_pixel_space_FiT.png" width="48%" alt="latent+pixel space FiT Pipeline">

  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/inference_pipelines/run_model_final_pipeline.png" width="48%" alt="multi adapter pipeline Pipeline">
</p> -->


# Results
These are the results obtained from run_model.py.

<p align="center">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_74/output_74a.png" width="19%" alt="First Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_74/output_74b.png" width="19%" alt="Second Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_74/output_74c.png" width="19%" alt="Third Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_74/output_74d.png" width="19%" alt="Fourth Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_74/output_74e.png" width="19%" alt="Fifth Image">
</p>

<p align="center">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_88/output_88a.png" width="19%" alt="First Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_88/output_88b.png" width="19%" alt="Second Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_88/output_88c.png" width="19%" alt="Third Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_88/output_88d.png" width="19%" alt="Fourth Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_88/output_88e.png" width="19%" alt="Fifth Image">
</p>

<p align="center">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_90/output_90a.png" width="19%" alt="First Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_90/output_90b.png" width="19%" alt="Second Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_90/output_90c.png" width="19%" alt="Third Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_90/output_90d.png" width="19%" alt="Fourth Image">
  <img src="https://github.com/anmol-c03/image_gen_hackathon/blob/main/images/results/output_90/output_90e.png" width="19%" alt="Fifth Image">
</p>




## Installation

1. Clone the repository:
```bash
git clone https://github.com/anmol-c03/image_gen_hackathon.git

```

2. Install dependencies:
```bash
pip install -r requirements.txt
```



## Usage

```bash
(If you have dataset as defined in run_model.py format)
!python run_model.py 

(If has images only)

!python inference/infer.py


```
<!-- 
## Project Structure

```bash
Structured_handwritten_data_extraction/                      # Project root
├── images/  
│   ├── original/
│   ├── resized/
│   └── extract_pdf/                 # Folder for images
├── model_doclayout/                 # Model-related files for layout processor
├── models/                          # Folder for storing YOLO model for text detection
|    |_bestline.pt                         
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── layout_processor.py
│   ├── text_processor.py
│   ├── text_recognition.py
│   └── correction_processor.py                      # Processing scripts
├── Table_extraction/                 # Table extraction module
│   ├── images/                       # Subfolder for table images
│   ├── __init__.py                   # Init file
│   ├── cell_coordinates.py           # Table cell detection script
│   ├── crop_table.py                 # Table cropping script
│   ├── main.py                       # Main script for table extraction
│   ├── ocr.py                        # OCR processing script
│   └── preprocess.py                 # Preprocessing script
├── txt/                              # Folder for extracted text files
│   ├── txt.1                         
│   ├── txt.2                         
│   ├── txt.3                         
├── utils/
│   ├── __init__.py
│   └── file_utils.py                          # Utility functions
├── .gitignore                        # Git ignore file
├── install.sh                        # Installation script
├── main_for_api.py                   # Main script for API integration
├── main.py                           # Main script
├── Readme.md                         # Project documentation
└── requirements.txt                   # Dependencies list
 -->


## Requirements

- Python 3.7+
- CUDA compatible GPU (recommended for faster processing)
- See requirements.txt  for required packages

