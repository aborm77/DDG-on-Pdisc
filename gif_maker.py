# -*- coding: utf-8 -*-
"""
@author: Ari Bormanis
Purpose: Create a gif from images
"""

import imageio.v3 as iio
import os


def make_gif(im_dir, gif_name, dur=135, gif_dir='gifs'):
    """Assemble all fig*.png files in im_dir into a gif saved to gif_dir/."""
    _, _, files = next(os.walk(im_dir))
    file_count = len(files)
    images = []
    for i in range(file_count):
        images.append(iio.imread(os.path.join(im_dir, f'fig{i}.png')))
    os.makedirs(gif_dir, exist_ok=True)
    iio.imwrite(os.path.join(gif_dir, f'{gif_name}.gif'), images, duration=dur, loop=0)


if __name__ == '__main__':
    # im_dir uses Windows path separators; change to 'figs/cut_bp1/' on Mac/Linux
    im_dir = 'figs\\cut_bp1\\'
    gif_name = 'Change_cut_bp1'
    dur = 135
    make_gif(im_dir, gif_name, dur)
