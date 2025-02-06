# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import numpy as np

import Pdisc_SG_vis as vis
from Pdisc_SG_classes import Sol_tree
import gc

        
    
        

if __name__ == '__main__':
    rad = False
    sep = False
    angle = False
    cutoff = True

    # Playing with a changing radius
    if rad:
        step = 0.1
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 2.5, np.round(1 + i*step,1), 0.1)
            jeff.bp1()
            # ploting with just base points
            vis.plot_grid_pdisc(jeff.base, save=True, f_name='rad/fig'+str(i))
            # plotting branch points as well
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, save=True, f_name='rad_bps/fig'+str(i))
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='rad_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()
            
        
    # Playing with changing sepearation
    if sep:
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 2.5, 3, 3 / (i+10) )
            jeff.bp1()
            # ploting with just base points
            vis.plot_grid_pdisc(jeff.base, save=True, f_name='sep/fig'+str(i))
            # plotting branch points as well
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, save=True, f_name='sep_bps/fig'+str(i))
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='sep_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()
    
    # Playing with the initial angle
    if angle:
        step = ((3 * np.pi)/8) / 40
        for i in range(41):
            jeff = Sol_tree(np.pi/2 - (i * step), 2.5, 3, 3 / 40 )
            jeff.bp1()
            # ploting with just base points
            vis.plot_grid_pdisc(jeff.base, save=True, f_name='angle/fig'+str(i))
            # plotting branch points as well
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, save=True, f_name='angle_bps/fig'+str(i))
            # # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='angle_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()
            
    # Playing with the cutoff angle
    if cutoff:
        step = (3 - 1.7) / 40
        for i in range(41):
            jeff = Sol_tree(np.pi/2, 1.7 + (i*step), 3, 3 / 40 )
            jeff.bp1()
            # ploting with just base points
            vis.plot_grid_pdisc(jeff.base, save=True, f_name='cut/fig'+str(i))
            # plotting branch points as well
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, save=True, f_name='cut_bps/fig'+str(i))
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='cut_bps_bndry/fig'+str(i))
            del jeff
            gc.collect()
                
            
