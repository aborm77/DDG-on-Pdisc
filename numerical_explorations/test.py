# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:40:03 2025

@author: Ari Bormanis
"""

import numpy as np

import Pdisc_SG_math as math_ddg
from multiprocessing import Pool



# given boundary data on two rays creates a Chebyshev net on the Poincare disk
# sol_grid stores x values of pts in [:,:,0], y values in [:,:,1], and the solution
# to the sine-Gordon equation in [:,:,2]
def grid_solve(xbd, ybd):
    rows = ybd.shape[0]
    cols = xbd.shape[0]
    sol_grid = np.zeros((rows, cols, 3))
    sol_grid[0,:,:] = xbd
    sol_grid[:,0,:] = ybd
    
    # Now creating cheby net
    for i in range(1, rows):
        for j in range(1, cols):
            z2 = sol_grid[i,j-1,:2]
            z0 = sol_grid[i-1,j-1,:2]
            z1 = sol_grid[i-1,j  ,:2]
            w2 = math_ddg.f(z2,-z0)
            w1 = math_ddg.f(z1,-z0)
            w12 = math_ddg.w12_comp(w1,w2)
            sol_grid[i,j,2] = math_ddg.angle(w1,w2)
            sol_grid[i,j,:2] = math_ddg.f(w12,z0)
            
    return sol_grid


# Returns 2x2 rotation matrix using angle phi
def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])


# Creates geodesic ray with specified lenght, angle, and sine-gordon solution data
def create_geo(arg,sep,npts,data):
    R = int(sep * npts)
    x = np.linspace(0,R,npts)
    x = np.tanh(x/2)
    y = np.zeros(npts)
    ax = np.vstack((x,y))
    if (arg != 0):
        ax = np.matmul(rot(arg), ax)
    ax = np.concatenate((ax,data), axis=0)
        
    return ax.T

if __name__ == '__main__':
    npts = 3
    R0 = 4
    phi0 = np.pi/2
    sep = R0 / npts
    
    phis = np.full((1,npts), phi0)
    
    ax1 = create_geo(0, sep, npts, phis)
    ax2 = create_geo(phi0, sep, npts, phis)
    ax3 = create_geo(phi0+0.4, sep, npts, phis)
    
    with Pool() as p:
        res = p.starmap(grid_solve, [(ax1,ax2),(ax2,ax3)])
    print(res)
    print(len(res))
    print(res[0].shape)