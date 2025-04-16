# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import numpy as np
import pyvista as pv
from numpy.linalg import norm

import Pdisc_SG_math as math_ddg
import Pdisc_SG_vis as vis


def angle(v1, v2):
    return np.arccos(np.dot(v1,v2) / (norm(v1) * norm(v2)))

class Basic_grid:
    def __init__(self, grid, children, bp_loc):
        self.grid = grid
        self.children = children
        self.bp_loc = bp_loc
        

"""
This class creates a cheby net on the p-disc given x- and y-boundary data
and a maximal geodesic radius
"""
class Sol_grid:
    def __init__(self, xbd, ybd, parent, r, ams):
        self.xbd = xbd
        self.ybd = ybd
        self.parent = parent
        self.r = r
        self.ams = ams
        self.rows = self.ybd.shape[0]
        self.cols = self.xbd.shape[0]
        self.r1 = xbd[-1,0]
        self.r2 = ybd[-1,1]
        self.npts = xbd.shape[0]

        self.bp_loc = None
        self.children = None
        
        self.grid = self.grid_solve()     
        rdiag = math_ddg.norm([self.grid[-1,-1,0], self.grid[-1,-1,1]])
        self.rmax = np.max([self.r1, self.r2, rdiag])
        
        
        self.angle_test()
        self.sep_test()
        
        
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
                w2 = math_ddg.f(z2,-z0)
                w1 = math_ddg.f(z1,-z0)
                w12 = math_ddg.w12_comp(w1,w2)
                # sol_grid[i+1,j+1,2] = math_ddg.angle(w1,w2)
                sol_grid[i,j,2] = math_ddg.angle(w1,w2)
                    
                sol_grid[i+1,j+1,:2] = math_ddg.f(w12,z0)
        return sol_grid
    
    
    def sep_test(self):
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                z0 =  self.grid[i  , j  , :2]
                z1 =  self.grid[i  , j+1, :2]
                z2 =  self.grid[i+1, j  , :2]
                z12 = self.grid[i+1, j+1, :2]
                
                # finding alph0
                w1 = math_ddg.f(z1,-z0)
                w2 = math_ddg.f(z2,-z0)
                alph0 = math_ddg.angle(w1,w2)
                
                # finding alph12
                ww1 = math_ddg.f(z1,-z12)
                ww2 = math_ddg.f(z2,-z12)
                alph12 = math_ddg.angle(ww1,ww2)
                
                # print(math_ddg.geo_dist(ww1))
                # print(math_ddg.geo_dist(ww2))
                
        
    def angle_test(self):
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                z0 =  self.grid[i  , j  , :2]
                z1 =  self.grid[i  , j+1, :2]
                z2 =  self.grid[i+1, j  , :2]
                z12 = self.grid[i+1, j+1, :2]
                
                # finding alph0
                w1 = math_ddg.f(z1,-z0)
                w2 = math_ddg.f(z2,-z0)
                alph0 = math_ddg.angle(w1,w2)
                
                # finding alph12
                ww1 = math_ddg.f(z1,-z12)
                ww2 = math_ddg.f(z2,-z12)
                alph12 = math_ddg.angle(ww1,ww2)
                
                # finding alph1
                h1 = math_ddg.f(z12,-z1)
                h2 = math_ddg.f(z0,-z1) 
                alph1 = math_ddg.angle(h1,h2)
                
                # finding alph2
                hh1 = math_ddg.f(z12,-z2)
                hh2 = math_ddg.f(z0,-z2) 
                alph2 = math_ddg.angle(hh1,hh2)
                
                a_diff1 = np.abs(alph0 - alph12)
                a_diff2 = np.abs(alph1 - alph2)
                
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
    def __init__(self, phi0, cutoff, r, sep):
        self.phi0 = phi0
        self.cutoff = cutoff
        self.r = r
        self.sep = sep
        self.npts = int(np.ceil(r / sep)) +1
        
        if (phi0 >= np.pi or phi0 <=0):
            print("Error: phi0 is not in the allowed range of (0,pi)")
            return -1
        if (phi0 > cutoff):
            print("Error: value of phi0 is greater than the cutoff value")
            print("branch point placement impossible")
            return -1
        
        xbd = self.create_geo(0   , phi0, self.npts)
        ybd = self.create_geo(phi0, phi0, self.npts)
        self.base = Sol_grid(xbd, ybd, 'base', self.r, 'ams')
        
        print('Initialized a solution tree with intial grid of size ' + str(self.npts) +'x' +  str(self.npts))
    
    # Creates geodesic ray with specified lenght, angle, and sine-gordon solution data
    def create_geo(self, arg, data, npts):
        r = self.sep * (npts-1)
        x = np.linspace(0, r, npts)
        x = np.tanh(x/2)
        y = np.zeros(npts)
        ax = np.vstack((x,y))
        if (arg != 0):
            ax = np.matmul(math_ddg.rot(arg), ax)
        phis = np.full((1, npts), data)
        ax = np.concatenate((ax,phis), axis=0)
            
        return ax.T
        
    
    # checks to see if the boundry of the sector lies outside the geodesic 
    # of radius self,r
    def bndry_check(self, sol_grid):
        xpts = sol_grid.xbd[:,:2]
        ypts = sol_grid.ybd[:,:2]
        pts = np.vstack((xpts,ypts))
        
        for pt in pts:
            if math_ddg.geo_dist(pt) < sol_grid.r:
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
        # print('bnd')
        # print(diffs)
        # print('bnd')
        
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
        if sol_grid.ams =='ams3':
            vs = self.bnd_find(sol_grid, rho_target)
            
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
        
        base_grid[:    , :vs + 1, :] = sol_grid.grid[:    , :vs + 1, :]
        base_grid[:us+1, vs+1:  , :] = sol_grid.grid[:us+1, vs+1:  , :]
        
            
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
        sect1 = Sol_grid(geo2, ax1, sol_grid, self.r, 'ams1')
        sect2 = Sol_grid(geo1, geo2, sol_grid, self.r, 'ams')
        sect3 = Sol_grid(ax3, geo1, sol_grid, self.r, 'ams3')
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
        
    
class Norm_tree:
    def __init__(self, sol_tree):
        self.sep = np.arccos(1 / np.cosh(sol_tree.sep)) 
        self.npts = sol_tree.npts
        self.norms = np.zeros((self.npts,self.npts,3))
        
        self.a_base = self.angle_clean(sol_tree.base)
        pl = pv.Plotter()
        self.grid_solve(sol_tree.base,pl)
        pl.add_axes()
        pl.show()
        
    def angle_clean(self, sol_grid, depth=0):
        a_grid = np.pi - np.copy(sol_grid.grid[:,:,2])
        ag = Basic_grid(a_grid, sol_grid.children, sol_grid.bp_loc)
        
        if (sol_grid.children != None):
            for child in sol_grid.children:
               self.angle_clean(child, depth=depth+1)
        
        if depth == 0:
            return ag
        
        
        
    def solve(self, us, vs, pl, b0=np.zeros(3), b1=np.zeros(3)):
        c = np.array([0,0,0])
        
        for j in range(vs):
            for i in range(us):
                if (i == 0 and j == 0):
                    if np.any(self.norms[0,0,:] != 0):
                        continue
                    
                    if np.any(b0 != 0):
                        n0 = b0
                        n1 = b1
                        
                    else:
                        n0 = np.array([0,0,1])
                        
                        e2 = np.array([0,1,0])
                        n1 = math_ddg.rod(n0, e2, self.sep)
                        n1 /= norm(n1)
                    
                    v1 = np.cross(n0,n1)
                    v2 = math_ddg.rod(v1, n0, self.a_base.grid[0,0])
                    v2 /= norm(v2)
                    n2 = math_ddg.rod(n0, v2, self.sep)
                    
                    vv12 = np.cross(n1,n2)
                    vv12 /= norm(vv12)
                    n12 = math_ddg.house(n0, vv12)
                    n12 /= norm(n12)
                    
                    self.norms[0,0,:] = n0
                    self.norms[0,1,:] = n1
                    self.norms[1,0,:] = n2
                    self.norms[1,1,:] = n12
                    
                    pl.add_arrows(c, n0, mag=1, color='blue')
                    pl.add_arrows(c, n1, mag=1, color='black')
                    pl.add_arrows(c, n2, mag=1, color='red')
                    pl.add_arrows(c, n12, mag=1, color='green')
                    
                elif (j==0): 
                    if np.any(self.norms[i+1, j, :] != 0):
                        continue
                    
                    n0 = self.norms[i,j,:]
                    n1 = self.norms[i,j+1,:]
                    
                    v1 = np.cross(n0,n1)
                    v2 = math_ddg.rod(v1, n0, self.a_base.grid[i,j])
                    v2 /= norm(v2)
                    n2 = math_ddg.rod(n0, v2, self.sep)
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math_ddg.house(n0, vv12)
                    n12 /= norm(n12)
                    
                    self.norms[i+1,   j, :] = n2
                    self.norms[i+1, j+1, :] = n12
                    
                    pl.add_arrows(c, n2, mag=1, color='cyan')
                    pl.add_arrows(c, n12, mag=1, color='purple')
                    
                elif (i==0):
                    if np.any(self.norms[i, j+1, :] != 0):
                        continue
                    
                    n0 = self.norms[i  , j]
                    n2 = self.norms[i+1, j]
                    
                    v1 = np.cross(n2, n0)
                    v2 = math_ddg.rod(v1, n0, -self.a_base.grid[i,j])
                    v2 /= norm(v2)
                    
                    n1 = math_ddg.rod(n0, v2, -self.sep)
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math_ddg.house(n0, vv12)
                    n12 /= norm(n12)
                    
                    self.norms[i  , j+1, :] = n1
                    self.norms[i+1, j+1, :] = n12
                    
                    pl.add_arrows(c, n1, mag=1, color='cyan')
                    pl.add_arrows(c, n12, mag=1, color='purple')
                    
                else: 
                    if np.any(self.norms[i+1, j+1, :] != 0):
                        continue
                    
                    n0 = self.norms[i  , j]
                    n1 = self.norms[i  , j+1]
                    n2 = self.norms[i+1, j]
                    
                    vv12 = np.cross(n1, n2)
                    vv12 /= norm(vv12)
                    n12 = math_ddg.house(n0, vv12)
                    n12 /= norm(n12)
                
                    self.norms[i+1, j+1, :] = n12
                    if j % 2 == 0:
                        pl.add_arrows(c, n12, mag=1, color='purple')
                    else:
                        pl.add_arrows(c, n12, mag=1, color='orange')   

        
    def grid_solve(self, sol_grid, pl, depth=0, b1=0, b2=0):
        
        if self.a_base.bp_loc == None:
            if depth == 0:
                self.solve(self.npts - 1, self.npts - 1, pl)
            else:
                self.solve(self.npts - 1, self.npts - 1, pl)
        else:
            us, vs = self.a_base.bp_loc
            if depth == 0:
                self.solve(us, self.npts - 1, pl)
                self.solve(self.npts - 1, vs, pl)
            else:
                pass
            
        if (sol_grid.children != None):
            for child in sol_grid.children:
               self.grid_solve(child, pl, depth=depth+1)
        
        if depth == 0:
            return 0
            
       
        
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
                    
                    
    def angle_test(self):
        # first checking to see if angles match what they should be in a_grid 
        quad_angles = np.zeros((self.npts - 1, self.npts - 1, 4))
        for i in range(self.npts - 1):
            for j in range(self.npts - 1):
                n0 =  self.norms[i  , j  , :]
                n1 =  self.norms[i  , j+1, :]
                n2 =  self.norms[i+1, j  , :]
                n12 = self.norms[i+1, j+1, :]
                
                v1 = np.cross(n1, n0)
                v2 = np.cross(n2, n0)
                a = angle(v1, v2) 
                a_diff = np.abs(angle(v1, v2) - self.a_base.grid[i,j])
                
                
                # print('Actual angle:', a)
                # print('a_grid angle:', self.a_base.grid[i,j])
                if a_diff > 1e-8:
                    print('Warning! Some of the angles are incorrect')
                    print('Angle at '+'('+str(i)+','+str(j)+')'+' is wrong')
                    print('Actual angle:', a)
                    print('a_grid angle:', self.a_base.grid[i,j])
                     
                
                v1 = np.cross(n1, n0)
                v2 = np.cross(n2, n0)
                a0 = angle(v1, v2)
                
                v1 = np.cross(n1, n0)
                v2 = np.cross(n1, n12)
                a1 = angle(v1, v2)
                
                v1 = np.cross(n2, n0)
                v2 = np.cross(n2, n12)
                a2 = angle(v1, v2)
                
                v1 = np.cross(n12, n1)
                v2 = np.cross(n12, n2)
                a12 = angle(v1, v2)
                
                quad_angles[i,j,:] = [a0, a1, a2, a12]
                A = - 2 * np.pi + (a0 + a1 + a2 + a12)
        
    


if __name__ == '__main__':
    # phi0 = np.pi/16
    # jeff = Sol_tree(phi0, 2.5, 3, 0.1)
    # jeff.bp2(3*phi0)
    step = 0.1
    i = 34
    jeff = Sol_tree(np.pi/2, 2.5, 2, 2/10)
    jeff.bp1(max_depth=1)
    vis.plot_grid_pdisc(jeff.base, plt_bps=True, plt_bds=True)
    norman = Norm_tree(jeff)
    # norman.sep_test()
    # norman.norm_test()
    # norman.angle_test()
    
    # vis.plot_grid_rho(jeff.base)
    




