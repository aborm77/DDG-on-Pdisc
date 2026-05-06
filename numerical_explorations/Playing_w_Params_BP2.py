# -*- coding: utf-8 -*-
"""
Parameter sweep scripts comparing bp1 and bp2 branch point algorithms.

Generates side-by-side figure sequences by sweeping over radius, separation,
initial angle, and cutoff. Run from the repo root:

    python numerical_explorations/Playing_w_Params_BP2.py

Author: Ari Bormanis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import vis
from classes import Sol_tree
import gc


if __name__ == '__main__':
    rad = True
    sep = False
    angle = False
    cutoff = False

    if rad:
        step = 0.1
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 2.5, np.round(1 + i*step, 1), 0.1)
            jeff2 = Sol_tree(np.pi/5, 2.5, np.round(1 + i*step, 1), 0.1)
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
            vis.Pdisc_plot(jeff1.base, save=True, f_name='rad_bp1/fig'+str(i))
            vis.Pdisc_plot(jeff2.base, save=True, f_name='rad_bp2/fig'+str(i))
            del jeff1, jeff2
            gc.collect()

    if sep:
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 2.5, 3, 3 / (i+10))
            jeff2 = Sol_tree(np.pi/5, 2.5, 3, 3 / (i+10))
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
            vis.Pdisc_plot(jeff1.base, save=True, f_name='sep_bp1/fig'+str(i))
            vis.Pdisc_plot(jeff2.base, save=True, f_name='sep_bp2/fig'+str(i))
            del jeff1, jeff2
            gc.collect()

    if angle:
        step = ((3 * np.pi)/16) / 40
        for i in range(41):
            jeff1 = Sol_tree(np.pi/4 - (i * step), 2.5, 3, 3 / 40)
            jeff2 = Sol_tree(np.pi/4 - (i * step), 2.5, 3, 3 / 40)
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
            vis.Pdisc_plot(jeff1.base, save=True, f_name='angle_bp1/fig'+str(i))
            vis.Pdisc_plot(jeff2.base, save=True, f_name='angle_bp2/fig'+str(i))
            del jeff1, jeff2
            gc.collect()

    if cutoff:
        step1 = (3 - 1.7) / 40
        step2 = 2 / 40
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 1.7 + (i*step1), 3, 3 / 40)
            jeff2 = Sol_tree(np.pi/5, 2, 3, 3 / 40)
            jeff1.bp1()
            jeff2.bp2(np.pi/5 * (1 + i * step2))
            vis.Pdisc_plot(jeff1.base, save=True, f_name='cut_bp1/fig'+str(i))
            vis.Pdisc_plot(jeff2.base, save=True, f_name='cut_bp2/fig'+str(i))
            del jeff1, jeff2
            gc.collect()
