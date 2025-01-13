# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import numpy as np
import matplotlib.pyplot as plt

import Pdisc_SG_math as math_ddg


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
            


class Sol_tree:
    def __init__(self, phi0, cutoff, r0, sep):
        self.phi0 = phi0
        self.cutoff = cutoff
        self.r0 = r0
        self.sep = sep
        self.npts = int(np.floor(r0 / sep))
        
        if (phi0 >= np.pi or phi0 <=0):
            print("Error: phi0 is not in the allowed range of (0,pi)")
            return -1
        if (phi0 > cutoff):
            print("Error: value of phi0 is greater than the cutoff value")
            print("branch point placement impossible")
            return -1
        
        xbd = self.create_geo(0   , phi0, self.npts)
        ybd = self.create_geo(phi0, phi0, self.npts)
        self.base = Sol_grid(xbd, ybd, 'base')
        
        print('Initialized a solution tree with intial grid of size ' + str(self.npts) +'x' +  str(self.npts))
    
    # Creates geodesic ray with specified lenght, angle, and sine-gordon solution data
    def create_geo(self, arg, data, npts):
        r = self.sep * npts
        x = np.linspace(0, r, npts)
        x = np.tanh(x/2)
        y = np.zeros(npts)
        ax = np.vstack((x,y))
        if (arg != 0):
            ax = np.matmul(math_ddg.rot(arg), ax)
        phis = np.full((1, npts), data)
        ax = np.concatenate((ax,phis), axis=0)
            
        return ax.T
    
    # place a branch point and return a solution grid representing the new L-shaped region
    def place_bp(self, sol_grid):
        nan_ignore = np.ma.array(sol_grid.grid[:,:,2], mask=np.isnan(sol_grid.grid[:,:,2]))
        if np.all(nan_ignore < self.cutoff):
            return None
    
        rows = sol_grid.grid.shape[0]
        cols = sol_grid.grid.shape[1]
        base_grid = np.full((rows, cols, 3), np.nan)
    
        # finding branch points
        us = 0
        vs = 0
        for i in range(rows):
            if sol_grid.grid[i,-1,2] >= self.cutoff:
                us = i-1
                break
        for i in range(cols):
            if sol_grid.grid[-1,i,2] >= self.cutoff:
                vs = i-1
                break
        
        if (vs < 0 or us < 0):
            print('Inappropriate boundary data, branch point placement impossible')
            return -1
        
        base_grid[:    , :us + 1, :] = sol_grid.grid[:    , :us + 1, :]
        base_grid[:vs+1, us+1:  , :] = sol_grid.grid[:vs+1, us+1:  , :]
        
        
        # Sometimes the above placement of a branch point won't work and we have
        # to decrease the index of the branch point like so
        # this ensures that our L-shaped region has values strictly less than
        # the cutoff
        # while ((np.any(base_grid[:,:,2] > self.cutoff)) and (us > 0 and vs > 0)):
        #     us -= 1
        #     vs -= 1
        #     base_grid = np.full((rows, cols, 3), np.nan)
        #     base_grid[:    , :us + 1, :] = sol_grid.grid[:    , :us + 1, :]
        #     base_grid[:vs+1, us + 1:  , :] = sol_grid.grid[:vs+1, us + 1:  , :]
        
        # if (us==0 or vs==0):
        #     print('Oh no')
        #     print(sol_grid.grid[:,:,2])
        #     plot_grid_rho(sol_grid)
        #     return None
            
        sol_grid.grid = base_grid
        sol_grid.bp_loc = (us,vs)
        
        return (us,vs)
    
    
    def create_sectors1(self, sol_grid):
        bp_loc = sol_grid.bp_loc
        us = bp_loc[0]
        vs = bp_loc[1]
        
        # print()
        # print(sol_grid.grid.shape)
        # print(bp_loc)
        # plot_grid_pdisc(sol_grid)
        # print()
        bp_val = sol_grid.grid[us,vs,2] / 3
        ax1_len = len(sol_grid.grid[us:,vs,:])
        ax3_len = len(sol_grid.grid[us,vs:,:])
        max_len = max(ax1_len,ax3_len)
        
        sect1 = np.zeros((ax1_len,ax1_len,3))
        sect2 = np.zeros((max_len,max_len,3))
        sect3 = np.zeros((ax3_len,ax3_len,3))
        
        ax1 = np.copy(sol_grid.grid[us:,vs,:])
        ax3 = np.copy(sol_grid.grid[us,vs:,:])
        ax1[:,2] = ax1[:,2] - 2 * bp_val
        ax3[:,2] = ax3[:,2] - 2 * bp_val
        
        z2 = sol_grid.grid[us+1, vs ,:2]
        z0 = sol_grid.grid[us  ,vs  ,:2]
        z1 = sol_grid.grid[us  ,vs+1,:2]
   
        w2 = math_ddg.f(z2,-z0)
        w1 = math_ddg.f(z1,-z0)
        arg1 = np.arctan2(w1[1],w1[0])
        arg2 = np.arctan2(w2[1],w2[0])
        
        # finding angles and creating geodesics
        phi1 = (2*arg1 + arg2) / 3
        phi2 = (arg1 + 2*arg2) / 3
        
        geo1 = self.create_geo(phi1, bp_val, max_len)
        geo2 = self.create_geo(phi2, bp_val, max_len)
        for i in range(max_len):
            geo1[i,:2] = math_ddg.f(geo1[i,:2],z0)
            geo2[i,:2] = math_ddg.f(geo2[i,:2],z0)
            
        # solving on new grids
        sect1 = Sol_grid(geo2, ax1, sol_grid)
        sect2 = Sol_grid(geo1, geo2, sol_grid)
        sect3 = Sol_grid(ax3, geo1, sol_grid)
        sol_grid.children = [sect1, sect2, sect3]
        
        return sect1, sect2, sect3
        
        
        
    def bp1(self, sol_grid=None):
        if sol_grid == None:
            sol_grid = self.base
        bp_loc = self.place_bp(sol_grid)
        if bp_loc == None or bp_loc == -1:
            return
        sect1, sect2, sect3 = self.create_sectors1(sol_grid)
        self.bp1(sol_grid=sect1)
        self.bp1(sol_grid=sect2)
        self.bp1(sol_grid=sect3)
        
        # us = bp_loc[0]
        # vs = bp_loc[1]
        # bp_val = self.base.grid[us,vs,2] / 3
        # ax1_len = len(self.base.grid[us:,vs,:])
        # ax3_len = len(self.base.grid[us,vs:,:])
        # max_len = max(ax1_len,ax3_len)
        
        # sect1 = np.zeros((ax1_len,ax1_len,3))
        # sect2 = np.zeros((max_len,max_len,3))
        # sect3 = np.zeros((ax3_len,ax3_len,3))
        
        # ax1 = np.copy(self.base.grid[us:,vs,:])
        # ax3 = np.copy(self.base.grid[us,vs:,:])
        # ax1[:,2] = ax1[:,2] - 2 * bp_val
        # ax3[:,2] = ax3[:,2] - 2 * bp_val
        
        # z2 = self.base.grid[us+1, vs ,:2]
        # z0 = self.base.grid[us  ,vs  ,:2]
        # z1 = self.base.grid[us  ,vs+1,:2]
   
        # w2 = math_ddg.f(z2,-z0)
        # w1 = math_ddg.f(z1,-z0)
        # arg1 = np.arctan2(w1[1],w1[0])
        # arg2 = np.arctan2(w2[1],w2[0])
        
        # # finding angles and creating geodesics
        # phi1 = (2*arg1 + arg2) / 3
        # phi2 = (arg1 + 2*arg2) / 3
        
        # geo1 = self.create_geo(phi1, bp_val, max_len)
        # geo2 = self.create_geo(phi2, bp_val, max_len)
        # for i in range(max_len):
        #     geo1[i,:2] = math_ddg.f(geo1[i,:2],z0)
        #     geo2[i,:2] = math_ddg.f(geo2[i,:2],z0)
            
        # # solving on new grids
        # sect1 = Sol_grid(geo2, ax1, self.base)
        # sect2 = Sol_grid(geo1, geo2, self.base)
        # sect3 = Sol_grid(ax3, geo1, self.base)
        # self.base.children = [sect1, sect2, sect3]
        
        # plot_grid_rho(self.base)
        # plot_grid_pdisc(self.base)
        
    
        
        
        
class Sol_grid:
    def __init__(self, xbd, ybd, parent):
        self.xbd = xbd
        self.ybd = ybd
        self.parent = parent
        self.r1 = xbd[-1,0]
        self.r2 = ybd[-1,1]
        self.npts = xbd.shape[0]

        self.bp_loc = None
        self.children = None
        self.grid = self.grid_solve()     
    
        
    # given boundary data on two rays creates a Chebyshev net on the Poincare disk
    # sol_grid stores x values of pts in [:,:,0], y values in [:,:,1], and the solution
    # to the sine-Gordon equation in [:,:,2]
    def grid_solve(self):
        rows = self.ybd.shape[0]
        cols = self.xbd.shape[0]
    
        sol_grid = np.zeros((rows,cols,3))
        sol_grid[0,:,:] = self.xbd
        sol_grid[:,0,:] = self.ybd
        
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
    


jeff = Sol_tree(np.pi/2, 2.5, 3, 0.3)
jeff.bp1()
# plot_grid_rho(jeff.base)
plot_grid_pdisc(jeff.base)


    
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
     
     ax1 = np.copy(base_grid[us:,vs,:])
     ax3 = np.copy(base_grid[us,vs:,:])
     ax1[:,2] = ax1[:,2] - 2 * bp_val
     ax3[:,2] = ax3[:,2] - 2 * bp_val
     
     z2 = base_grid[us+1, vs ,:2]
     z0 = base_grid[us  ,vs  ,:2]
     z1 = base_grid[us  ,vs+1,:2]

     w2 = f(z2,-z0)
     w1 = f(z1,-z0)
     arg1 = np.arctan2(w1[1],w1[0])
     arg2 = np.arctan2(w2[1],w2[0])
     
     # finding angles and creating geodesics
     phi1 = (2*arg1 + arg2) / 3
     phi2 = (arg1 + 2*arg2) / 3
     
     geo1 = create_geo(phi1, sep, max_len, np.full((1,max_len),bp_val))
     geo2 = create_geo(phi2, sep, max_len, np.full((1,max_len),bp_val))
     for i in range(max_len):
         geo1[i,:2] = f(geo1[i,:2],z0)
         geo2[i,:2] = f(geo2[i,:2],z0)
         
     # solving on new grids
     sect1 = grid_solve(geo2, ax1)
     sect2 = grid_solve(geo1, geo2)
     sect3 = grid_solve(ax3, geo1)
     
     return sect1, sect2, sect3
     
    

# Make the boundary geodesics
npts = 100
R0 = 4
phi0 = np.pi/2
sep = R0 / npts
cutoff = 3.1

phis = np.full((1,npts), phi0)

# ax1 = create_geo(0, sep, npts, phis)
# ax2 = create_geo(phi0, sep, npts, phis)

# # Getting solution
# sol_grid = grid_solve(ax1, ax2)
# base_grid, bp_loc = place_bp(sol_grid, cutoff)
# sect1, sect2, sect3 = create_sectors(base_grid, bp_loc, sep)



