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
