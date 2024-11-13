# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import numpy as np
import matplotlib.pyplot as plt


# Returns 2x2 rotation matrix using angle phi
def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])


# Applies the mobius function described in the distribtued branch points paper
# The function has been rationalized so we can just apply it to real vectors
def f(pt,z0):
    dot = 1 + pt[0] * z0[0] + pt[1] * z0[1]
    cross = pt[1] * z0[0] - pt[0] * z0[1]
    denom = dot**2 + cross**2
    num = pt + z0
    x = dot * num[0] + cross * num[1]
    y = dot * num[1] - cross * num[0]
    return np.array([x,y]) / denom


# Computes the last vertex in the quadralaterial like in the branch points paper
def w12_comp(w1, w2):
    num = w1 + w2
    denom = 1 + np.sqrt((w1[0]*w2[0] - w1[1]*w2[1])**2 + (w1[0]*w2[1] + w1[1]*w2[0])**2)
    return num / denom


# Gets the approx angle for the Sin-Gordon equation
def norm(w):
    return np.sqrt(w[0]**2 + w[1]**2 )


# Computes the angle betwen two vectors using the dot product
def angle(w1,w2):
    dot = w1[0]*w2[0] + w1[1]*w2[1]
    return np.arccos(dot / (norm(w1) * norm(w2)))


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


# given boundary data on two rays creates a Chebyshev net on the Poincare disk
# sol_grid stores x values of pts in [:,:,0], y values in [:,:,1], and the solution
# to the sine-Gordon equation in [:,:,2]
def grid_solve(ax1,ax2):
    rows = ax2.shape[0]
    cols = ax1.shape[0]

    sol_grid = np.zeros((rows,cols,3))
    sol_grid[0,:,:] = ax1
    sol_grid[:,0,:] = ax2
    
    # Now creating cheby net
    for i in range(1, rows):
        for j in range(1, cols):
            z2 = sol_grid[i,j-1,:2]
            z0 = sol_grid[i-1,j-1,:2]
            z1 = sol_grid[i-1,j  ,:2]
            w2 = f(z2,-z0)
            w1 = f(z1,-z0)
            w12 = w12_comp(w1,w2)
            sol_grid[i,j,2] = angle(w1,w2)
            sol_grid[i,j,:2] = f(w12,z0)
            
    return sol_grid
    
# place a branch point and return a solution grid representing the new L-shaped region
def place_bp(sol_grid, cutoff):
    npts = sol_grid[:,:,0].shape[0]
    base_grid = np.zeros((npts,npts,3))

    # finding branch points
    us = 0
    vs = 0
    for i in range(npts):
        if sol_grid[-1,i,2] >= cutoff:
            us = i-1
        if sol_grid[i,-1,2] >= cutoff:
            vs = i-1
        if (vs != 0 and us != 0):
            break
    
    if (us == 0 and vs == 0):
        print('No branch points were needed \n returning original grid')
        return sol_grid, -1
    if (vs < 0 or us < 0):
        print('Inappropriate boundary data, branch point placement impossible')
        return sol_grid, -1
    
    base_grid[:    , :us + 1, :] = sol_grid[:    , :us + 1, :]
    base_grid[:vs+1, us+1:  , :] = sol_grid[:vs+1, us+1:  , :]
    
    # Sometimes the above placement of a branch point won't work and we have
    # to decrease the index of the branch point like so
    # this ensures that our L-shaped region has values strictly less than
    # the cutoff
    while ((np.any(base_grid[:,:,2] > cutoff)) and (us > 0 and vs > 0)):
        us -= 1
        vs -= 1
        base_grid = np.zeros((npts,npts,3))
        base_grid[:    , :us + 1, :] = sol_grid[:    , :us + 1, :]
        base_grid[:vs+1, us + 1:  , :] = sol_grid[:vs+1, us + 1:  , :]
    
    if (us==0 or vs==0):
        print("Warning, degenerate case: us = 0 or vs = 0")
    
    return base_grid, (us,vs)

# given a base_grid this creates the three new sectors stemming from it 
def create_sectors(base_grid, bp_loc, sep):
     us = bp_loc[0]
     vs = bp_loc[1]
     bp_val = base_grid[us,vs,2] / 3
     ax1_len = len(base_grid[us:,vs,:])
     ax3_len = len(base_grid[us,vs:,:])
     max_len = max(ax1_len,ax3_len)
     
     sect1 = np.zeros((ax1_len,ax1_len,3))
     sect2 = np.zeros((max_len,max_len,3))
     sect3 = np.zeros((ax3_len,ax3_len,3))
     
     ax1 = base_grid[us:,vs,:]
     ax3 = base_grid[us,vs:,:]
     
     z2 = base_grid[us+1, vs ,:2]
     z0 = base_grid[us  ,vs  ,:2]
     z1 = base_grid[us  ,vs+1,:2]
     w2 = f(z2,-z0)
     w1 = f(z1,-z0)
     e1 = np.array([1,0])
     arg1 = angle(e1,w1)
     arg2 = angle(e1,w2)
     
     # finding angles and creating geodesics
     phi1 = (2*arg1 + arg2) / 3
     phi2 = (arg1 + 2*arg2) / 3
     geo1 = create_geo(phi1, sep, max_len, np.full((1,max_len),bp_val))
     geo2 = create_geo(phi2, sep, max_len, np.full((1,max_len),bp_val))
     # transforming geodesics back to z0
     for i in range(max_len):
         geo1[i,:2] = f(geo1[i,:2],z0)
         geo2[i,:2] = f(geo2[i,:2],z0)
         
     # solving on new grids
     sect1 = grid_solve(geo2, ax1)
     sect2 = grid_solve(geo1, geo2)
     sect3 = grid_solve(ax3,geo1)
     
     
     return sect1, sect2, sect3
     
    

# Make the boundary geodesics
npts = 30
R0 = 6
phi0 = 1.1
sep = R0 / npts

# x1 = np.linspace(0,R0,npts)
# x1 = np.tanh(x1/2)
# y1 = np.zeros(npts)

# ax1 = np.vstack((x1,y1))
# ax2 = np.matmul(rot(phi0), ax1)

phis = np.full((1,npts), phi0)
# ax1 = np.concatenate((ax1,phis), axis=0)
# ax2 = np.concatenate((ax2,phis), axis=0)

ax1 = create_geo(0, sep, npts, phis)
ax2 = create_geo(phi0, sep, npts, phis)

# Getting solution
sol_grid = grid_solve(ax1, ax2)
base_grid, bp_loc = place_bp(sol_grid, 2.5)
sect1, sect2, sect3 = create_sectors(base_grid, bp_loc, sep)


# function to plot both the points on the pdisc and the rho (as a surface graph)
def plot_grid(sol_grid, R, plots):
    npts = sol_grid[:,:,0].shape[0]
    if (plots=='both' or plots=='pdisc'):
        plt.figure(1)
        plt.scatter(sol_grid[:,:,0],sol_grid[:,:,1], alpha=0.5)
        plt.xlim(-0.1, np.tanh(R/2)+0.1)
        plt.ylim(-0.1, np.tanh(R/2)+0.1)
        plt.gca().set_aspect('equal')
    if (plots=='both' or plots=='sg'):
        plt.figure(2)
        x = np.linspace(0,R,npts)
        y = np.linspace(0,R,npts)
        x, y = np.meshgrid(x,y)
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, sol_grid[:,:,2], alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('rho')

# Plotting
com = 'pdisc'
plot_grid(base_grid, R0, com)
plot_grid(sect1, R0, com)
plot_grid(sect3, R0, com)
plot_grid(sect2, R0, com)



