from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--celeba_input_dir', type=str, help='', required=True)
parser.add_argument('--celeba_output_dir', type=str, help='', required=True)

args = parser.parse_args()

files = os.listdir(args.celeba_input_dir)

if not os.path.exists(args.celeba_output_dir):
    os.mkdir(args.celeba_output_dir)

new_width = 108
new_height = 108

for file in files:
    f = os.path.join(args.celeba_input_dir, file)
    im = Image.open(f)
    width, height = im.size

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.save(os.path.join(args.celeba_output_dir, file))