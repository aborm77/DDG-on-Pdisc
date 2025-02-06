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
    step = 0.1
    # ploting with just base points
    for i in range(41):
        jeff = Sol_tree(np.pi/2, 2.5, np.round(1 + i*step,1), 0.1)
        jeff.bp1()
        vis.plot_grid_pdisc(jeff.base, save=True, f_name='rad/fig'+str(i))
        del jeff
        gc.collect()
        
    # plotting branch points as well
    # for i in range(41):
    #     jeff = Sol_tree(np.pi/2, 2.5, np.round(1 + i*step,1), 0.1)
    #     jeff.bp1()
    #     vis.plot_grid_pdisc(jeff.base, plt_bps=True, save=True, f_name='rad_bps/fig'+str(i))
    #     del jeff
    #     gc.collect()
        
    # plotting branch points and sector boundries
    # for i in range(41):
    #     jeff = Sol_tree(np.pi/2, 2.5, np.round(1 + i*step,1), 0.1)
    #     jeff.bp1()
    #     vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, save=True, f_name='rad_bps_bndry/fig'+str(i))
    #     del jeff
    #     gc.collect()
        
        
    # jeff = Sol_tree(np.pi/2, 2.5, 3, 0.1)
    # jeff.bp1()
    # vis.plot_grid_pdisc(jeff.base, save=True, f_name='rad_no_bps/jeff')
    # plt.savefig('figs/test.png', dpi=400)
    # vis.plot_grid_rho(jeff.base)




