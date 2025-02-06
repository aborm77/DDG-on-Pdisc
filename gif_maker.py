# -*- coding: utf-8 -*-
"""
@author: Ari Bormanis
Purpose: Create a gif from images
"""

import imageio.v3 as iio
import os

# If you are on windows remember to put '\\' for every \ in the directory
im_dir = 'figs\\rad\\'
gif_name = 'Change_rad'
dur = 135

_, _, files = next(os.walk(im_dir))
file_count = len(files)
images = []
for i in range(file_count):
  images.append(iio.imread(im_dir + 'fig'+ str(i)+ '.png'))
iio.imwrite(gif_name+'.gif', images, duration = dur, loop = 0)