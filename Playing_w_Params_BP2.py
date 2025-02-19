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
    rad = True
    sep = False
    angle = False
    cutoff = False

    # Playing with a changing radius
    if rad:
        step = 0.1
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 2.5, np.round(1 + i*step,1), 0.1)
            jeff2 = Sol_tree(np.pi/5, 2.5, np.round(1 + i*step,1), 0.1)
            
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
            
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff1.base, plt_bps=True, plt_bds=True, save=True, f_name='rad_bp1/fig'+str(i))
            vis.plot_grid_pdisc(jeff2.base, plt_bps=True, plt_bds=True, save=True, f_name='rad_bp2/fig'+str(i))
            
            del jeff1
            del jeff2
            gc.collect()
            
        
    # Playing with changing sepearation
    if sep:
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 2.5, 3, 3 / (i+10) )
            jeff2 = Sol_tree(np.pi/5, 2.5, 3, 3 / (i+10) )
            
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
            
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff1.base, plt_bps=True, plt_bds=True, save=True, f_name='sep_bp1/fig'+str(i))
            vis.plot_grid_pdisc(jeff2.base, plt_bps=True, plt_bds=True, save=True, f_name='sep_bp2/fig'+str(i))
            
            del jeff1
            del jeff2
            gc.collect()
    
    # Playing with the initial angle
    if angle:
        step = ((3 * np.pi)/16) / 40
        for i in range(41):
            jeff1 = Sol_tree(np.pi/4 - (i * step), 2.5, 3, 3 / 40 )
            jeff2 = Sol_tree(np.pi/4 - (i * step), 2.5, 3, 3 / 40 )
            
            jeff1.bp1()
            jeff2.bp2(3 * np.pi/5)
  
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff1.base, plt_bps=True, plt_bds=True, save=True, f_name='angle_bp1/fig'+str(i))
            vis.plot_grid_pdisc(jeff2.base, plt_bps=True, plt_bds=True, save=True, f_name='angle_bp2/fig'+str(i))
            
            del jeff1
            del jeff2
            gc.collect()
            
    # Playing with the cutoff angle and rho target
    if cutoff:
        step1 = (3 - 1.7) / 40
        step2 = 2 / 40
        for i in range(41):
            jeff1 = Sol_tree(np.pi/5, 1.7 + (i*step1), 3, 3 / 40 )
            jeff2 = Sol_tree(np.pi/5, 2, 3, 3 / 40 )
            
            jeff1.bp1()
            jeff2.bp2(np.pi/5 * (1 + i * step2))
            
            # plotting branch points and sector boundries
            vis.plot_grid_pdisc(jeff1.base, plt_bps=True, plt_bds=True, save=True, f_name='cut_bp1/fig'+str(i))
            vis.plot_grid_pdisc(jeff2.base, plt_bps=True, plt_bds=True, save=True, f_name='cut_bp2/fig'+str(i))
            
            del jeff1
            del jeff2
            gc.collect()
