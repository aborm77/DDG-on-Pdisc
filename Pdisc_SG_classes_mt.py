# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: Trying to use paralleization, this unfortunaley just seems to slow 
things down
"""

import numpy as np

import Pdisc_SG_math as math_ddg
import Pdisc_SG_vis as vis
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





"""
This class creates a cheby net on the p-disc given x- and y-boundary data
and a maximal geodesic radius
"""
class Sol_grid:
    def __init__(self, grid, parent):
        self.grid = grid 
        self.xbd = grid[0,:,:]
        self.ybd = grid[:,0,:]
        self.parent = parent
        self.rows = self.ybd.shape[0]
        self.cols = self.xbd.shape[0]
        self.r1 = self.xbd[-1,0]
        self.r2 = self.ybd[-1,1]

        self.bp_loc = None
        self.children = None


"""
This class preforms the branch point placement process and in doing so creates
a tree of Sol_grid objects
"""
class Sol_tree:
    def __init__(self, phi0, cutoff, r, sep):
        self.phi0 = phi0
        self.cutoff = cutoff
        self.r = r
        self.sep = sep
        self.npts = int(np.ceil(r / sep))
        
        if (phi0 >= np.pi or phi0 <=0):
            print("Error: phi0 is not in the allowed range of (0,pi)")
            return -1
        if (phi0 > cutoff):
            print("Error: value of phi0 is greater than the cutoff value")
            print("branch point placement impossible")
            return -1
        
        xbd = self.create_geo(0   , phi0, self.npts)
        ybd = self.create_geo(phi0, phi0, self.npts)
        grid = grid_solve(xbd, ybd)
        self.base = Sol_grid(grid, 'base')
        
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
        
    
    # checks to see if the boundry of the sectore lies outside the geodesic 
    # of radius self,r
    def bndry_check(self, sol_grid):
        xpts = sol_grid.xbd[:,:2]
        ypts = sol_grid.ybd[:,:2]
        pts = np.vstack((xpts,ypts))
        
        for pt in pts:
            if math_ddg.geo_dist(pt) < self.r:
                return False
        return True
    
    
    # place a branch point and return a solution grid representing the new L-shaped region
    def place_bp(self, sol_grid):
        nan_ignore = np.ma.array(sol_grid.grid[:,:,2], mask=np.isnan(sol_grid.grid[:,:,2]))
        if np.all(nan_ignore < self.cutoff):
            return None
    
        rows = sol_grid.rows
        cols = sol_grid.cols
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
        
        if (vs < 0) or (us < 0):
            print('Error: inappropriate boundary data, branch point placement impossible')
            return -1
        if (vs == 0) or (us == 0):
            print('Warning: branch point was placed on a boundary')
        
        base_grid[:    , :vs + 1, :] = sol_grid.grid[:    , :vs + 1, :]
        base_grid[:us+1, vs+1:  , :] = sol_grid.grid[:us+1, vs+1:  , :]
        
            
        sol_grid.grid = base_grid
        sol_grid.bp_loc = (us,vs)
        
        if self.bndry_check(sol_grid):
            return
        
        return (us,vs)
    
    
    # Given a solution grid with a branch point already placed, creates the 
    # three sectors that spawn from it
    def create_sectors1(self, sol_grid):
        bp_loc = sol_grid.bp_loc
        us = bp_loc[0]
        vs = bp_loc[1]
        
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
        with Pool() as p:
            grids = p.starmap(grid_solve, [(geo2, ax1), (geo1, geo2), (ax3, geo1)])
        # grid1 = grid_solve(geo2, ax1)
        # grid2 = grid_solve(geo1, geo2)
        # grid3 = grid_solve(ax3, geo1)
        sect1 = Sol_grid(grids[0], sol_grid)
        sect2 = Sol_grid(grids[1], sol_grid)
        sect3 = Sol_grid(grids[2], sol_grid)
        sol_grid.children = [sect1, sect2, sect3]
        
        return sect1, sect2, sect3
        
        
    # preforms the branch point algorithm described in the paper
    def bp1(self, sol_grid=None, depth=0, max_depth=None):
        if sol_grid == None:
            sol_grid = self.base
        if (max_depth != None and depth == max_depth):
            return 
        bp_loc = self.place_bp(sol_grid)
        if (bp_loc == None) or (bp_loc == -1):
            return 
        
        sect1, sect2, sect3 = self.create_sectors1(sol_grid)
        self.bp1(sol_grid=sect1, depth=depth+1, max_depth=max_depth)
        self.bp1(sol_grid=sect2, depth=depth+1, max_depth=max_depth)
        self.bp1(sol_grid=sect3, depth=depth+1, max_depth=max_depth)
        
        if depth == 0:
            print('Done, graphing now')
        
    
        

if __name__ == '__main__':
    r = 4
    jeff = Sol_tree(np.pi/2, 2, r, 0.4/2)
    jeff.bp1()
    vis.plot_grid_pdisc(jeff.base, r, plt_bps=True)
    vis.plot_grid_rho(jeff.base)




