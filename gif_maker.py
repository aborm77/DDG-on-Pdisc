# -*- coding: utf-8 -*-
"""
@author: Ari Bormanis
Purpose: Create a gif from images
"""

import imageio.v3 as iio
import os
from PIL import Image, ImageChops


def make_gif(im_dir, gif_name, dur=135, gif_dir='gifs'):
    """Assemble all fig*.png files in im_dir into a gif saved to gif_dir/."""
    _, _, files = next(os.walk(im_dir))
    file_count = len(files)
    images = []
    for i in range(file_count):
        images.append(iio.imread(os.path.join(im_dir, f'fig{i}.png')))
    os.makedirs(gif_dir, exist_ok=True)
    iio.imwrite(os.path.join(gif_dir, f'{gif_name}.gif'), images, duration=dur, loop=0)


def _autocrop_box(gif, padding=10):
    """Return the crop box that removes whitespace, consistent across all frames."""
    left, upper, right, lower = gif.width, gif.height, 0, 0
    white = Image.new('RGB', (gif.width, gif.height), (255, 255, 255))
    for i in range(gif.n_frames):
        gif.seek(i)
        bbox = ImageChops.difference(gif.convert('RGB'), white).getbbox()
        if bbox:
            left  = min(left,  bbox[0])
            upper = min(upper, bbox[1])
            right = max(right, bbox[2])
            lower = max(lower, bbox[3])
    return (
        max(0, left  - padding),
        max(0, upper - padding),
        min(gif.width,  right  + padding),
        min(gif.height, lower  + padding),
    )


def combine_gifs_horizontal(gif_paths, output_path, gap=4, autocrop=None):
    """Combine GIFs side by side into a single GIF.

    All input GIFs should have the same frame count. Frame duration is taken
    from the first GIF. A white gap of `gap` pixels is inserted between each.

    autocrop: None/False (no crop), True (crop all), or list of bool per GIF.
    """
    gifs = [Image.open(p) for p in gif_paths]
    n_frames = min(g.n_frames for g in gifs)

    if autocrop is None or autocrop is False:
        autocrop = [False] * len(gifs)
    elif autocrop is True:
        autocrop = [True] * len(gifs)

    crop_boxes = [_autocrop_box(g) if do_crop else None
                  for g, do_crop in zip(gifs, autocrop)]

    def _cropped_size(gif, box):
        if box:
            return (box[2] - box[0], box[3] - box[1])
        return (gif.width, gif.height)

    sizes = [_cropped_size(g, cb) for g, cb in zip(gifs, crop_boxes)]
    total_w = sum(s[0] for s in sizes) + gap * (len(gifs) - 1)
    max_h   = max(s[1] for s in sizes)

    gifs[0].seek(0)
    dur = gifs[0].info.get('duration', 150)

    frames = []
    for i in range(n_frames):
        canvas = Image.new('RGB', (total_w, max_h), (255, 255, 255))
        x = 0
        for gif, crop_box, (w, h) in zip(gifs, crop_boxes, sizes):
            gif.seek(i)
            frame = gif.convert('RGB')
            if crop_box:
                frame = frame.crop(crop_box)
            canvas.paste(frame, (x, (max_h - h) // 2))
            x += w + gap
        frames.append(canvas.convert('P', palette=Image.ADAPTIVE, colors=256))

    for g in gifs:
        g.close()

    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   loop=0, duration=dur, optimize=True)


if __name__ == '__main__':
    # im_dir uses Windows path separators; change to 'figs/cut_bp1/' on Mac/Linux
    # im_dir = 'figs\\cut_bp1\\'
    # gif_name = 'Change_cut_bp1'
    # dur = 135
    # make_gif(im_dir, gif_name, dur)
    combine_gifs_horizontal(
        ['gifs/grow_pdisc.gif', 'gifs/grow_sphere.gif', 'gifs/grow_surf.gif'],
        'gifs/grow_combined.gif',
        autocrop=True
    )
