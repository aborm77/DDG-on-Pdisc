# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import matplotlib.pyplot as plt
import numpy as np


def geo_circ(r):
    t = np.linspace(0, np.pi/2, 50)
    x = np.tanh(r / 2) * np.cos(t)
    y = np.tanh(r / 2) * np.sin(t)
    return x,y
    

# function to plot the points on the pdisc
def plot_grid_pdisc(sol_grid, depth=0, fig=None, ax=None, bd_sect=False, plt_bps=False, plt_colors=False, plt_bds=False, save=False, f_name=None, pt_size=10):
    if (sol_grid == None):
        return
    if (depth == 0):
        fig, ax = plt.subplots()
        # r1 = sol_grid.r1
        # r2 = sol_grid.r2
        # plt.xlim(-0.1, r1+0.13)
        # plt.ylim(-0.1, r2+0.13)
        rmax = sol_grid.rmax
        plt.xlim(-0.1, rmax+0.14)
        plt.ylim(-0.1, rmax+0.14)
        # plt.xlim(-0.1, 1.10)
        # plt.ylim(-0.1, 1.10)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_aspect('equal')
        circ_x, circ_y = geo_circ(sol_grid.r)
        ax.plot(circ_x, circ_y, c='black', zorder=0)
        
    # avoids ploting points on the boundry of sectors twice
    if bd_sect:
        x = sol_grid.grid[1:,1:,0]
        y = sol_grid.grid[1:,1:,1]
    else:
        x = sol_grid.grid[:,:,0]
        y = sol_grid.grid[:,:,1]
        # avoids ploting branch points multiple times
        if depth != 0:
            x[0,0] = np.nan
            y[0,0] = np.nan
            
        
    # cycle through default colors each sector if plt_colors=True
    if plt_colors:
        ax.scatter(x, y, alpha=0.5,  zorder=1, s=pt_size)
    else:
        ax.scatter(x, y, alpha=0.5, c='C0',  zorder=1, s=pt_size)
    
    if (sol_grid.children != None):
        i = 0
        for child in sol_grid.children:
            if i % 2 != 0:
                plot_grid_pdisc(child, depth+1, fig, ax, plt_bps=plt_bps, plt_colors=plt_colors, plt_bds=plt_bds, pt_size=pt_size)
            else:
                plot_grid_pdisc(child, depth+1, fig, ax, bd_sect=True, plt_bps=plt_bps, plt_colors=plt_colors, plt_bds=plt_bds, pt_size=pt_size)
            i += 1
            
    # plots branch points if plt_bps = True
    if (sol_grid.bp_loc != None and plt_bps):
        loc1 = sol_grid.bp_loc[0] 
        loc2 = sol_grid.bp_loc[1] 
        ax.scatter(sol_grid.grid[loc1,loc2,0], sol_grid.grid[loc1,loc2,1], c='red', zorder=4, s=pt_size)
        
    # plots the boundries of sectors if plt_bds = True
    if plt_bds:
        xbd = sol_grid.xbd[:,:2]
        ybd = sol_grid.ybd[:,:2]
        ax.plot(xbd[:,0], xbd[:,1], c='orange', zorder=3)
        ax.plot(ybd[:,0], ybd[:,1], c='orange', zorder=3)
        
    # saves the image if save=True
    if depth == 0 and save:
        if f_name == None:
            plt.savefig('figs/test.png', dpi=100)
        else:
            plt.savefig('figs/'+f_name+'.png', dpi=100)
        plt.close()
    if depth == 0 and not save:
        plt.show()
    
# function to plot the values of rho (as a surface graph)     
def plot_grid_rho(sol_grid, depth=0, fig=None, ax=None):
    if (sol_grid == None):
        return
    if (depth == 0):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('rho')
        
    x = sol_grid.grid[:,:,0]
    y = sol_grid.grid[:,:,1]
    rho = sol_grid.grid[:,:,2]
    ax.plot_surface(x, y, rho, alpha=0.5, color='C'+str(depth))
    
    if (sol_grid.children != None):
        for child in sol_grid.children:
            plot_grid_rho(child, depth+1, fig, ax)
            


