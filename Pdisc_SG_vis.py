# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import matplotlib.pyplot as plt


# function to plot the points on the pdisc
def plot_grid_pdisc(sol_grid, depth=0, fig=None, ax=None):
    if (sol_grid == None):
        return
    if (depth == 0):
        fig, ax = plt.subplots()
        r1 = sol_grid.r1
        r2 = sol_grid.r2
        plt.xlim(-0.1, r1+0.1)
        plt.ylim(-0.1, r2+0.1)
        ax.set_aspect('equal')
        
    x = sol_grid.grid[:,:,0]
    y = sol_grid.grid[:,:,1]
    ax.scatter(x, y, alpha=0.5)
    # if sol_grid.bp_loc != None:
    #     loc1 = sol_grid.bp_loc[0] 
    #     loc2 = sol_grid.bp_loc[1] 
    #     ax.scatter(sol_grid.grid[loc1,loc2,0], sol_grid.grid[loc1,loc2,1], alpha=0.5)
    
    if (sol_grid.children != None):
        for child in sol_grid.children:
            plot_grid_pdisc(child, depth+1, fig, ax)
            
    
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
    ax.plot_surface(x, y, rho, alpha=0.5)
    
    if (sol_grid.children != None):
        for child in sol_grid.children:
            plot_grid_rho(child, depth+1, fig, ax)
            


