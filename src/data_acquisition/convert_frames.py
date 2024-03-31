"""
script to convert tiff images to png images

inputs:
    input_dir (path to folder containing .tif images) -> string
    output_dir (path to folder to save .png images) -> string

outputs:
    folder containing converted png images

Copyright (C) 2024  Saifullah Ijaz

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Saifullah Ijaz
date: 03/03/2024
"""

from imports import *

# inputs
input_dir = 'C:/Users/saifu/source/repos/Forest-SLAM/forest_data/1018_dalsa_garden_short/1018_garden_short_imgs/00/c54d7a'
output_dir = 'C:/Users/saifu/source/repos/Forest-SLAM/forest_data/1018_dalsa_garden_short/1018_garden_short_imgs/00/c54d7a_png'

# create output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through input folder
for img in os.listdir(input_dir):
    if img.endswith('.tif') or img.endswith('.tiff'):
        # open tiff image
        tiff_img = Image.open(os.path.join(input_dir, img))

        # convert to png
        png_img = tiff_img.convert('RGB')

        # get filename
        filename, _ = os.path.splitext(img)

        # save tiff image in output folder
        png_img.save(os.path.join(output_dir, filename + '.png'))
