# -*- coding: utf-8 -*-
"""
Parameter sweep scripts for the bp1 branch point algorithm.

Generates figure sequences by sweeping over radius, separation, initial angle,
and cutoff. Run from the repo root:

    python numerical_explorations/Playing_w_Params.py

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
    rad = False
    sep = False
    angle = False
    cutoff = True

    if rad:
        step = 0.1
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 2.5, np.round(1 + i*step, 1), 0.1)
            jeff.bp1()
            vis.Pdisc_plot(jeff.base, plt_bps=False, plt_bds=False, save=True, f_name='rad/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=False, save=True, f_name='rad_bps/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='rad_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()

    if sep:
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 2.5, 3, 3 / (i+10))
            jeff.bp1()
            vis.Pdisc_plot(jeff.base, plt_bps=False, plt_bds=False, save=True, f_name='sep/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=False, save=True, f_name='sep_bps/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='sep_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()

    if angle:
        step = ((3 * np.pi)/8) / 40
        for i in range(41):
            jeff = Sol_tree(np.pi/2 - (i * step), 2.5, 3, 3 / 40)
            jeff.bp1()
            vis.Pdisc_plot(jeff.base, plt_bps=False, plt_bds=False, save=True, f_name='angle/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=False, save=True, f_name='angle_bps/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='angle_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()

    if cutoff:
        step = (3 - 1.7) / 40
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 1.7 + (i*step), 3, 3 / 40)
            jeff.bp1()
            vis.Pdisc_plot(jeff.base, plt_bps=False, plt_bds=False, save=True, f_name='cut/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=False, save=True, f_name='cut_bps/fig'+str(i))
            vis.Pdisc_plot(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='cut_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()
