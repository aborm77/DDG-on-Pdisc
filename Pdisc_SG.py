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


# given boundary data on two rays creates a Chebyshev net on the Poincare disk
def grid_solve(ax1,ax2):
    npts = ax1.shape[1]
    sol_grid = np.zeros((npts,npts,3))
    # Adding bndry data for R^2 mesh
    sol_grid[0,:,0] = ax1[0,:]
    sol_grid[0,:,1] = ax1[1,:]
    sol_grid[:,0,0] = ax2[0,:]
    sol_grid[:,0,1] = ax2[1,:]
    # Adding bndry data for rho
    sol_grid[0,:,2] = ax1[2,:]
    sol_grid[:,0,2] = ax2[2,:]
    


# Make the boundary geodesics
npts = 50
R0 = 3
phi0 = np.pi/2
sep = R0 / npts

x1 = np.linspace(0,R0,npts)
x1 = np.tanh(x1/2)
y1 = np.zeros(npts)

ax1 = np.vstack((x1,y1))
ax2 = np.matmul(rot(phi0), ax1)


# Generating the rest of the solution on the first sector
# sol_grid1 stores x values of pts in [:,:,0] and y values in [0,0,:]
sol_grid = np.zeros((npts,npts,3))

# Adding bndry data for R^2 mesh
sol_grid[0,:,0] = ax1[0,:]
sol_grid[0,:,1] = ax1[1,:]
sol_grid[:,0,0] = ax2[0,:]
sol_grid[:,0,1] = ax2[1,:]
# Adding bndry data for rho
sol_grid[0,:,2] = phi0
sol_grid[:,0,2] = phi0



for i in range(1, npts):
    for j in range(1, npts):
        z2 = sol_grid[i,  j-1,:2]
        z0 = sol_grid[i-1,j-1,:2]
        z1 = sol_grid[i-1,j  ,:2]
        w2 = f(z2,-z0)
        w1 = f(z1,-z0)
        w12 = w12_comp(w1,w2)
        sol_grid[i,j,2] = angle(w1,w2)
        sol_grid[i,j,:2] = f(w12,z0)
        


plt.figure(1)
plt.scatter(sol_grid[:,:,0],sol_grid[:,:,1], alpha=0.5)
plt.xlim(-0.1, np.tanh(R0/2)+0.1)
plt.ylim(-0.1, np.tanh(R0/2)+0.1)
plt.gca().set_aspect('equal')

plt.figure(2)
x = np.linspace(0,R0,npts)
y = np.linspace(0,R0,npts)
x, y = np.meshgrid(x,y)

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, sol_grid[:,:,2], alpha=0.5)
 
# Set plot properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('rho')
# ax.view_init(30, -10)


