# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""
import numpy as np
from numpy.linalg import norm

import Pdisc_SG_math as math
import Pdisc_SG_vis as vis


"""
This class creates a cheby net on the Poincare disk given x- and y-boundary data
and a maximal geodesic radius. All points created in this way are stored as 
numpy arrays
"""
class Sol_grid:
    def __init__(self, xbd, ybd, parent, R, ams, test=True):
        self.xbd = xbd
        self.ybd = ybd
        self.parent = parent
        self.R = R
        self.ams = ams
        self.rows = self.ybd.shape[0]
        self.cols = self.xbd.shape[0]
        self.r1 = xbd[-1,0]
        self.r2 = ybd[-1,1]
        self.npts = xbd.shape[0]

        self.bp_loc = None
        self.children = None
        self.old_grid = None
        
        self.grid = self.grid_solve()     
        
        # determining the maximum distances in a sector for the purpose of plotting
        rdiag = math.norm([self.grid[-1,-1,0], self.grid[-1,-1,1]])
        self.rmax = np.max([self.r1, self.r2, rdiag])
        
        if test:
            self.angle_test()
        
        
    # given boundary data on two rays creates a Chebyshev net on the Poincare disk
    # sol_grid stores x values of pts in [:,:,0], y values in [:,:,1], and the solution
    # to the sine-Gordon equation in [:,:,2]
    def grid_solve(self):
        sol_grid = np.full((self.rows, self.cols, 3), np.nan)
        sol_grid[0,:,:2] = self.xbd[:,:2]
        sol_grid[:,0,:2] = self.ybd[:,:2]
        
        # Now creating cheby net
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                z2 = sol_grid[i+1, j  , :2]
                z0 = sol_grid[i  , j  , :2]
                z1 = sol_grid[i  , j+1, :2]
                w2 = math.f(z2,-z0)
                w1 = math.f(z1,-z0)
                w12 = math.w12_comp(w1,w2)
                sol_grid[i,j,2] = math.angle(w1,w2)
                sol_grid[i+1,j+1,:2] = math.f(w12,z0)
        return sol_grid
    
                
    # This tests to make sure we are actually constucting rhombi
    def angle_test(self):
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                z0 =  self.grid[i  , j  , :2]
                z1 =  self.grid[i  , j+1, :2]
                z2 =  self.grid[i+1, j  , :2]
                z12 = self.grid[i+1, j+1, :2]
                
                # finding a0
                w1 = math.f(z1,-z0)
                w2 = math.f(z2,-z0)
                a0 = math.angle(w1,w2)

                # finding a12
                ww1 = math.f(z1,-z12)
                ww2 = math.f(z2,-z12)
                a12 = math.angle(ww1,ww2)

                # finding a1
                h1 = math.f(z12,-z1)
                h2 = math.f(z0,-z1)
                a1 = math.angle(h1,h2)

                # finding a2
                hh1 = math.f(z12,-z2)
                hh2 = math.f(z0,-z2)
                a2 = math.angle(hh1,hh2)

                a_diff1 = np.abs(a0 - a12)
                a_diff2 = np.abs(a1 - a2)
                
                if (a_diff1 > 1e-8) or (a_diff2 > 1e-8):
                    print(a_diff1)
                    print(a_diff2)
                    print('Warning! Some of the angles are incorrect')
                    print('Issue found on quad with n0 at '+'('+str(i)+','+str(j)+')')
        


"""
This class preforms the branch point placement process and in doing so creates
a tree of Sol_grid objects
"""
class Sol_tree:
    def __init__(self, phi0, cutoff, R, sep):
        if (phi0 >= np.pi or phi0 <=0):
            print("Error: phi0 is not in the allowed range of (0,pi)")
            return -1
        if (phi0 > cutoff):
            print("Error: value of phi0 is greater than the cutoff value")
            print("branch point placement impossible")
            return -1
        
        self.phi0 = phi0
        self.cutoff = cutoff
        self.R = R
        self.sep = sep
        self.npts = int(np.ceil(R / sep)) +1
        
        xbd = self.create_geo(0   , phi0, self.npts)
        ybd = self.create_geo(phi0, phi0, self.npts)
        self.base = Sol_grid(xbd, ybd, 'base', self.R, 'ams')
        
        print('Initialized a solution tree with intial grid of size ' + str(self.npts) +'x' +  str(self.npts))
    
    # Creates geodesic ray with specified length, angle, and Sine-Gordon solution data
    def create_geo(self, phi, rho_bd, npts):
        ray_len = self.sep * (npts-1)
        x = np.linspace(0, ray_len, npts)
        x = np.tanh(x/2)
        y = np.zeros(npts)
        ax = np.vstack((x,y))
        if (phi != 0):
            ax = np.matmul(math.rot(phi), ax)
        rho_arr = np.full((1, npts), rho_bd)
        ax = np.concatenate((ax, rho_arr), axis=0)
            
        return ax.T
        
    
    # checks to see if the boundry of the sector lies outside the geodesic 
    # of radius self,r
    def bndry_check(self, sol_grid):
        xpts = sol_grid.xbd[:,:2]
        ypts = sol_grid.ybd[:,:2]
        pts = np.vstack((xpts,ypts))
        
        for pt in pts:
            if math.geo_dist(pt) < sol_grid.R:
                return False
        return True
    
    
    # place a branch point and return a solution grid representing the new L-shaped region
    def place_bp1(self, sol_grid):
        nan_ignore = np.ma.array(sol_grid.grid[:,:,2], mask=np.isnan(sol_grid.grid[:,:,2]))
        if np.all(nan_ignore < self.cutoff):
            return None
    
        rows = sol_grid.rows
        cols = sol_grid.cols
        base_grid = np.full((rows, cols, 3), np.nan)
        
        # finding branch points
        us = 0
        vs = 0
        for i in range(rows-1):
            if sol_grid.grid[i,-2,2] >= self.cutoff:
                us = i-1
                break
        for i in range(cols-1):
            if sol_grid.grid[-2,i,2] >= self.cutoff:
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
    
    
    # Finds the first node with asymptotic angle closest to rho_target 
    # Searches starting from the origin and returns the largest index
    # where the boundary data is less than rho_target 
    # Current research suggests setting rho_target = 3 phi0
    def bnd_find(self, sol_grid, rho_target):
        if sol_grid.ams == 'ams1':
            bd = sol_grid.ybd
        if sol_grid.ams == 'ams3':
            bd = sol_grid.xbd
        diffs = np.full((len(bd),), rho_target)
        diffs = diffs - bd[:,2]
        
        # this is looking for the node on the boundary with rho value closest 
        # to rho_target without exceeding rho_target
        for i in range(len(diffs)):
            if diffs[i] < 0:
                return i - 1
            
        # if this exacutes then the whole boundary has rho data < rho_target
        return -1

    
    # Perfroms very similarly to bnd_find but this time searches the diagonal
    # for the closest value to rho_target
    def diag_find(self, sol_grid, rho_target):
        diag = np.diagonal(sol_grid.grid[:,:,2])
        diffs = np.full((len(diag),), rho_target)
        diffs = diffs - diag
        
        for i in range(len(diffs)):
            if diffs[i] < 0:
                return i - 1
            
        return -1
        
    
    # This implaments the branch point placement in the new procedure
    def place_bp2(self, sol_grid, rho_target):
        us = 0
        vs = 0
        if sol_grid.ams =='ams':
            loc = self.diag_find(sol_grid, rho_target)
            us = loc
            vs = loc
        if sol_grid.ams =='ams1':
            us = self.bnd_find(sol_grid, rho_target)
            sol_grid.old_grid = np.copy(sol_grid.grid)
        if sol_grid.ams =='ams3':
            vs = self.bnd_find(sol_grid, rho_target)
            sol_grid.old_grid = np.copy(sol_grid.grid)
            
        # if this happens there is no need to place a branch point on the 
        # boundary since all boundary data is < rho_target
        if (us == -1) or (vs == -1):
            return
        
        if (us == 0) and (vs == 0):
            print('Warning, branch point placed at origin')
            return
        
        rows = sol_grid.rows
        cols = sol_grid.cols
        base_grid = np.full((rows, cols, 3), np.nan)
        

        
        if not (us == 0 or vs == 0):
            base_grid[:    , :vs + 1, :] = sol_grid.grid[:    , :vs + 1, :]
            base_grid[:us+1, vs+1:  , :] = sol_grid.grid[:us+1, vs+1:  , :]
        if vs == 0:
            base_grid[:us + 1, :, :] = sol_grid.grid[:us + 1, :, :]  
        if us == 0:
            base_grid[:, :vs + 1, :] = sol_grid.grid[:, :vs + 1, :]   
        
            
        sol_grid.grid = base_grid
        sol_grid.bp_loc = (us,vs)
        
        # still returning if our current grid has boundaries outside our 
        # geodesic radius
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
        
        if vs != 0 and us!=0:
            ax1 = np.copy(sol_grid.grid[us:,vs,:])
            ax3 = np.copy(sol_grid.grid[us,vs:,:])
        if vs == 0:
            ax1 = np.copy(sol_grid.old_grid[us:,vs,:])
            ax3 = np.copy(sol_grid.grid[us,vs:,:])
        if us == 0:
            ax1 = np.copy(sol_grid.grid[us:,vs,:])
            ax3 = np.copy(sol_grid.old_grid[us,vs:,:])
            
        
        ax1[:,2] = ax1[:,2] - 2 * bp_val
        ax3[:,2] = ax3[:,2] - 2 * bp_val
        
        if type(sol_grid.old_grid) != type(None):
            z2 = sol_grid.old_grid[us+1, vs ,:2]
            z0 = sol_grid.old_grid[us  ,vs  ,:2]
            z1 = sol_grid.old_grid[us  ,vs+1,:2]
        else: 
            z2 = sol_grid.grid[us+1, vs ,:2]
            z0 = sol_grid.grid[us  ,vs  ,:2]
            z1 = sol_grid.grid[us  ,vs+1,:2]
   
        w2 = math.f(z2,-z0)
        w1 = math.f(z1,-z0)
        arg1 = np.arctan2(w1[1],w1[0])
        arg2 = np.arctan2(w2[1],w2[0])
        
        # finding angles and creating geodesics
        phi1 = (2*arg1 + arg2) / 3
        phi2 = (arg1 + 2*arg2) / 3
        
        
        geo1 = self.create_geo(phi1, bp_val, max_len)
        geo2 = self.create_geo(phi2, bp_val, max_len)
        for i in range(max_len):
            geo1[i,:2] = math.f(geo1[i,:2],z0)
            geo2[i,:2] = math.f(geo2[i,:2],z0)
        
        # solving on new grids
        sect1 = Sol_grid(geo2, ax1, sol_grid, self.R, 'ams1')
        sect2 = Sol_grid(geo1, geo2, sol_grid, self.R, 'ams')
        sect3 = Sol_grid(ax3, geo1, sol_grid, self.R, 'ams3')
        sol_grid.children = [sect1, sect2, sect3]
        
        
        return sect1, sect2, sect3

        
        
    # preforms the branch point algorithm described in the paper
    def bp1(self, sol_grid=None, depth=0, max_depth=None):
        
        if sol_grid == None:
            sol_grid = self.base
        if (max_depth != None and depth == max_depth):
            return 
        bp_loc = self.place_bp1(sol_grid)
        if (bp_loc == None) or (bp_loc == -1):
            return 
        
        sect1, sect2, sect3 = self.create_sectors1(sol_grid)
        self.bp1(sol_grid=sect1, depth=depth+1, max_depth=max_depth)
        self.bp1(sol_grid=sect2, depth=depth+1, max_depth=max_depth)
        self.bp1(sol_grid=sect3, depth=depth+1, max_depth=max_depth)
        
        if depth == 0:
            print('Done placing branch points')
            
    # preforms the new experimental branch point algorithm 
    def bp2(self, rho_target, sol_grid=None, depth=0, max_depth=None):
        if sol_grid == None:
            sol_grid = self.base
        if (max_depth != None and depth == max_depth):
            return 
        bp_loc = self.place_bp2(sol_grid, rho_target)
        if bp_loc == None:
            return 
        
        sect1, sect2, sect3 = self.create_sectors1(sol_grid)
        self.bp2(rho_target, sol_grid=sect1, depth=depth+1, max_depth=max_depth)
        self.bp2(rho_target, sol_grid=sect2, depth=depth+1, max_depth=max_depth)
        self.bp2(rho_target, sol_grid=sect3, depth=depth+1, max_depth=max_depth)
        
        if depth == 0:
            print('Done placing branch points')


# Class used for getting rid of the nodes outside of the geodesic radius 
class Mask_grid:
    def __init__(self, sol_grid, parent):
        self.grid = np.copy(sol_grid.grid)
        self.cols = sol_grid.cols
        self.rows = sol_grid.rows
        self.R = sol_grid.R

        self.grid_mask()
        # true at pts (i,j) that are not equal to nan (i.e. the pts we want to graph)
        self.mask = np.logical_not(np.isnan(self.grid[:,:,0]))


    # setting points outside the geodesic radius eqaul to nan
    def grid_mask(self):
        for i in range(self.rows):
            for j in range(self.cols):
                pt = self.grid[i,j,:2]

                if not np.any(np.isnan(pt)):
                    dist = math.geo_dist(pt)
                    if dist > self.R:
                        self.grid[i,j,:] = np.nan


class Norm_grid:
    def __init__(self, sol_grid, parent, sep, b0=np.zeros(3), b1=np.zeros(3), b2=np.zeros(3), test=True):
        
        self.angles = np.pi - np.copy(sol_grid.grid[:,:,2])
        
        self.bp_loc = sol_grid.bp_loc
        self.npts = sol_grid.npts
        self.rows = sol_grid.rows
        self.cols = sol_grid.cols
        
        self.parent = parent
        self.children = None
        
        self.sep = sep
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        
        self.norms = np.zeros((self.rows,self.cols,3))
        
        self.grid_solve()
        
        if test:
            self.angle_test()
        
    def solve(self, us, vs):
        
        for j in range(vs):
            for i in range(us):
                if (i == 0 and j == 0):
                    if np.any(self.norms[0,0,:] != 0):
                        continue
                    
                    if np.any(self.b1 != 0):
                        n0 = self.b0
                        n1 = self.b1     
                    if np.any(self.b2 != 0):
                        n0 = self.b0
                        n2 = self.b2
                    if np.all(self.b0==0):
                        n0 = np.array([0,0,1])
                        e2 = np.array([0,1,0])
                        n1 = math.rod(n0, e2, self.sep)
                        n1 /= norm(n1)
                    
                    if np.any(self.b1 != 0) or np.all(self.b0==0):
                        v1 = np.cross(n0,n1)
                        v2 = math.rod(v1, n0, self.angles[0,0])
                        v2 /= norm(v2)
                        n2 = math.rod(n0, v2, self.sep)
                        
                        vv12 = np.cross(n1,n2)
                        vv12 /= norm(vv12)
                        n12 = math.house(n0, vv12)
                        n12 /= norm(n12)
                    else:
                        v1 = np.cross(n2, n0)
                        v2 = math.rod(v1, n0, -self.angles[0,0])
                        v2 /= norm(v2)
                        
                        n1 = math.rod(n0, v2, -self.sep)
                        
                        vv12 = np.cross(n1, n2)
                        vv12 /= norm(vv12)
                        n12 = math.house(n0, vv12)
                        n12 /= norm(n12)
        
                        
                    
                    self.norms[0,0,:] = n0
                    self.norms[0,1,:] = n1
                    self.norms[1,0,:] = n2
                    self.norms[1,1,:] = n12
                    
                elif (j==0): 
                    if np.any(self.norms[i+1, j, :] != 0):
                        continue
                    
                    n0 = self.norms[i,j,:]
                    n1 = self.norms[i,j+1,:]
                    
                    v1 = np.cross(n0,n1)
                    v2 = math.rod(v1, n0, self.angles[i,j])
                    v2 /= norm(v2)
                    n2 = math.rod(n0, v2, self.sep)
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math.house(n0, vv12)
                    n12 /= norm(n12)
                    
                    self.norms[i+1,   j, :] = n2
                    self.norms[i+1, j+1, :] = n12
                    
                elif (i==0):
                    if np.any(self.norms[i, j+1, :] != 0):
                        continue
                    
                    n0 = self.norms[i  , j]
                    n2 = self.norms[i+1, j]
                    
                    v1 = np.cross(n2, n0)
                    v2 = math.rod(v1, n0, -self.angles[i,j])
                    v2 /= norm(v2)
                    
                    n1 = math.rod(n0, v2, -self.sep)
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math.house(n0, vv12)
                    n12 /= norm(n12)
                    
                    self.norms[i  , j+1, :] = n1
                    self.norms[i+1, j+1, :] = n12
                    
                else: 
                    if np.any(self.norms[i+1, j+1, :] != 0):
                        continue
                    
                    n0 = self.norms[i  , j]
                    n1 = self.norms[i  , j+1]
                    n2 = self.norms[i+1, j]
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math.house(n0, vv12)
                    n12 /= norm(n12)
                
                    self.norms[i+1, j+1, :] = n12

    def grid_solve(self):
        if self.bp_loc == None:
            self.solve(self.rows - 1, self.cols - 1)
        else:
            us, vs = self.bp_loc
            # dealing with a degenerate but very possible case
            if us == 0 and vs==0:
                # just solving for the boundary
                self.solve(1, self.cols - 1)
                self.solve(self.rows - 1, 1)
                self.norms[1,1:,:] = 0
                self.norms[1:,1,:] = 0

            self.solve(us, self.cols - 1)
            self.solve(self.rows - 1, vs)

    def angle_test(self):
        # first checking to see if angles match what they should be in a_grid 
        quad_angles = np.zeros((self.rows - 1, self.cols - 1, 4))
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                n0 =  self.norms[i  , j  , :]
                n1 =  self.norms[i  , j+1, :]
                n2 =  self.norms[i+1, j  , :]
                n12 = self.norms[i+1, j+1, :]
                if np.any(n0 != 0) and np.any(n1 != 0) and np.any(n2 != 0) and np.any(n12 != 0):
                    v1 = np.cross(n1, n0)
                    v2 = np.cross(n2, n0)
                    a = math.angle(v1, v2) 
                    a_diff = np.abs(math.angle(v1, v2) - self.angles[i,j])
                    
    
                    if a_diff > 1e-8:
                        print('Warning! Some of the angles are incorrect')
                        print('Angle at '+'('+str(i)+','+str(j)+')'+' is wrong')
                        print('Actual angle:', a)
                        print('a_grid angle:', self.angles[i,j])
                         
                    
                    v1 = np.cross(n1, n0)
                    v2 = np.cross(n2, n0)
                    a0 = math.angle(v1, v2)
                    
                    v1 = np.cross(n1, n0)
                    v2 = np.cross(n1, n12)
                    a1 = math.angle(v1, v2)
                    
                    v1 = np.cross(n2, n0)
                    v2 = np.cross(n2, n12)
                    a2 = math.angle(v1, v2)
                    
                    v1 = np.cross(n12, n1)
                    v2 = np.cross(n12, n2)
                    a12 = math.angle(v1, v2)
                    
                    quad_angles[i,j,:] = [a0, a1, a2, a12]
                    # Uses Gauss-Bonet to calculate the area of a quad can be
                    # useful to see how the quad areas are changing
                    # A = - 2 * np.pi + (a0 + a1 + a2 + a12)
        
    
class Norm_tree:
    def __init__(self, sol_tree):
        self.sep = np.arccos(1 / np.cosh(sol_tree.sep)) 
        self.npts = sol_tree.npts
        self.norms_base = Norm_grid(sol_tree.base, 'base', self.sep)
        
        self.create_tree(sol_tree.base, self.norms_base)
        

    def create_tree(self, sol_grid, parent, depth=0):
        if sol_grid.bp_loc == None or sol_grid.children == None:
            return
        
        us, vs = sol_grid.bp_loc
        sol_children = sol_grid.children
        
            
        if us !=0: 
            ng3 = Norm_grid(sol_children[2], parent, self.sep, b0=parent.norms[us,vs,:], b1=parent.norms[us,vs+1,:])
            ng2 = Norm_grid(sol_children[1], parent, self.sep, b0=ng3.norms[0,0,:], b1=ng3.norms[1,0,:])
            ng1 = Norm_grid(sol_children[0], parent, self.sep, b0=ng2.norms[0,0,:], b1=ng2.norms[1,0,:])
        else:
            ng1 = Norm_grid(sol_children[0], parent, self.sep, b0=parent.norms[us,vs,:], b2=parent.norms[us+1,vs,:])
            ng2 = Norm_grid(sol_children[1], parent, self.sep, b0=ng1.norms[0,0,:], b2=ng1.norms[0,1,:])
            ng3 = Norm_grid(sol_children[2], parent, self.sep, b0=ng2.norms[0,0,:], b2=ng2.norms[0,1,:])
        
        norm_children = [ng1, ng2, ng3]
        parent.children = norm_children
        
        for i in range(3):
            self.create_tree(sol_children[i], norm_children[i], depth=depth+1)

        
    def sep_test(self):
        for i in range(self.npts - 1):
            for j in range(self.npts - 1):
                n0 =  self.norms[i  , j  , :]
                n1 =  self.norms[i  , j+1, :]
                n2 =  self.norms[i+1, j  , :]
                n12 = self.norms[i+1, j+1, :]
                
                sep1 = np.arccos(np.dot(n0,n2) / (norm(n0) * norm(n2))) - self.sep
                sep2 = np.arccos(np.dot(n0,n1) / (norm(n0) * norm(n1))) - self.sep
                sep3 = np.arccos(np.dot(n1,n12) / (norm(n1) * norm(n12))) - self.sep
                sep4 = np.arccos(np.dot(n2,n12) / (norm(n2) * norm(n12))) - self.sep
                
                tot_sep = abs(sep1) + abs(sep2) + abs(sep3) + abs(sep4)
                if (tot_sep > 1e-8):
                    print('Warning! Some of the seperations are incorrect')
                    print('Issue found on quad with n0 at '+'('+str(i)+','+str(j)+')')
                    print()
                    
                
    def norm_test(self):
        for i in range(self.npts - 1):
            for j in range(self.npts - 1):
                n0 =  self.norms[i  , j  , :]
                n1 =  self.norms[i  , j+1, :]
                n2 =  self.norms[i+1, j  , :]
                n12 = self.norms[i+1, j+1, :]
                
                diff1 = np.abs(norm(n0) - 1)
                diff2 = np.abs(norm(n1) - 1)
                diff3 = np.abs(norm(n2) - 1)
                diff4 = np.abs(norm(n12) - 1)
                
                if (diff1 + diff2 + diff3 + diff4) > 1e-8:
                    print("Error! Some of the normals are not unit length")
                    print('Issue found on quad with n0 at '+'('+str(i)+','+str(j)+')')
                    
                    
class Surf_grid:
    def __init__(self, norm_grid, parent, b0=np.full(3,np.nan), b1=np.full(3,np.nan), b2=np.full(3,np.nan), check=False):
        self.rows = norm_grid.rows
        self.cols = norm_grid.cols
        self.norms = norm_grid.norms
        self.bp_loc = norm_grid.bp_loc
        
        self.check = check
        if check:
            temp = np.copy(self.norms)
            self.norms = np.zeros((self.cols, self.rows,3))
            self.norms[:,:,0] = temp[:,:,0].T
            self.norms[:,:,1] = temp[:,:,1].T
            self.norms[:,:,2] = temp[:,:,2].T
            temp1 = self.rows
            self.rows = self.cols
            self.cols = temp1
            if self.bp_loc != None:
                self.bp_loc = self.bp_loc[::-1]
            
        
        self.grid = np.full((self.rows, self.cols,3),np.nan)
        
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        
        self.parent = parent
        self.children = None
        
        self.grid_solve()
        
    def solve(self, us, vs):
        
        for i in range(us):
            for j in range(vs):
                n0 =  self.norms[i  , j  ,:]
                n1 =  self.norms[i  , j+1,:]
                n2 =  self.norms[i+1, j  ,:]
                n12 = self.norms[i+1, j+1,:]
                
                if i == 0 and j ==0:
                    if not (np.any(np.isnan(self.b0))):
                        r0 = self.b0    
                    else:
                        r0 = np.zeros(3)
                        
                    if not (np.any(np.isnan(self.b1))):
                        r1 = self.b1
                    else:
                        r1 = r0 + np.cross(n1,n0)
                        
                    if not (np.any(np.isnan(self.b1))):
                        r2 = self.b2
                    else:
                        r2 = r0 - np.cross(n2,n0)
                   
                    r12 = r1 - np.cross(n12,n1)
                    
                    self.grid[i  , j  ,:] = r0
                    self.grid[i  , j+1,:] = r1
                    self.grid[i+1, j  ,:] = r2
                    self.grid[i+1, j+1,:] = r12
                    
                elif j == 0:
                    r0 = self.grid[i,j,:]
                    r2 = r0 - np.cross(n2,n0)
                    r12 = r2 + np.cross(n12,n2)
                    
                    self.grid[i+1, j  ,:] = r2
                    self.grid[i+1, j+1,:] = r12
                    
                elif i == 0:
                    r0 = self.grid[i,j,:]
                    r1 = r0 + np.cross(n1,n0)
                    r12 = r1 - np.cross(n12,n1)
                    
                    self.grid[i  , j+1,:] = r1
                    self.grid[i+1, j+1,:] = r12
                    
                else:
                    r1 = self.grid[i, j+1,:]
                    r12 = r1 - np.cross(n12,n1)
                    
                    self.grid[i+1, j+1,:] = r12
                    
    def grid_solve(self):
        if self.bp_loc == None:
            self.solve(self.rows - 1, self.cols - 1)
        else:
            us, vs = self.bp_loc
            self.solve(us, self.cols - 1)
            self.solve(self.rows - 1, vs)
            
                    
class Surf_tree:
    def __init__(self, norm_tree):
        self.npts = norm_tree.npts
        self.base = Surf_grid(norm_tree.norms_base, 'base')    
        self.create_tree(norm_tree.norms_base, self.base)
        
    def create_tree(self, norm_grid, parent, depth=0):
        if norm_grid.bp_loc == None or norm_grid.children == None:
            return
        
        pc = parent.check
        
        if pc:
            us, vs = norm_grid.bp_loc[::-1]
        else:
            us, vs = norm_grid.bp_loc
        sol_children = norm_grid.children
        
        pc = parent.check
        
        sg3 = Surf_grid(sol_children[2], parent, b0=parent.grid[us,vs,:], check= pc)
        sg2 = Surf_grid(sol_children[1], parent, b0=parent.grid[us,vs,:], check= not pc)
        sg1 = Surf_grid(sol_children[0], parent, b0=parent.grid[us,vs,:], check= pc)
        
        surf_children = [sg1, sg2, sg3]
        parent.children = surf_children
        
        for i in range(3):
            self.create_tree(sol_children[i], surf_children[i], depth=depth+1)


if __name__ == '__main__':
    # Talk params bp1
    phi0 = np.pi/3
    cutoff = 2.4
    R = 4
    sep = 0.1
    jeff = Sol_tree(phi0, cutoff, R, sep)
    jeff.bp1()
    
    
    # Talk params bp2
    # phi0 = np.pi/6
    # cutoff = 3 * phi0
    # R = 2.5
    # sep = 0.1
    # jeff = Sol_tree(phi0, cutoff, R, sep)
    # jeff.bp2(3*phi0)
    
    
    vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True, uv_lines=True)
    
    norman = Norm_tree(jeff)
    # vis.Arc_plot(norman.norms_base, depth_dis=True, plt_bps=True, plt_bds=True)
   
    sherman = Surf_tree(norman)
    vis.Surf_plot(sherman.base)
    