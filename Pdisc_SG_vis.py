# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: DDG solving of the Sine-Gordon equation on the Poincare disk
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def geo_circ(r):
    t = np.linspace(0, np.pi/2, 50)
    x = np.tanh(r / 2) * np.cos(t)
    y = np.tanh(r / 2) * np.sin(t)
    return x,y
    

# function to plot the points on the pdisc
def plot_grid_pdisc(sol_grid, depth=0, fig=None, ax=None, bd_sect=False, 
                    plt_bps=False, plt_colors=False, plt_bds=False, save=False, 
                    f_name=None, uv_lines=False, pt_size=10, rev=False):
    if (sol_grid == None):
        return
    if (depth == 0):
        fig, ax = plt.subplots()
        rmax = sol_grid.rmax
        plt.xlim(-0.1, rmax+0.14)
        plt.ylim(-0.1, rmax+0.14)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_aspect('equal')
        circ_x, circ_y = geo_circ(sol_grid.R)
        ax.plot(circ_x, circ_y, c='green', zorder=0)
        
    # avoids ploting points on the boundry of sectors twice
    if bd_sect and not uv_lines:
        x = sol_grid.grid[1:,1:,0]
        y = sol_grid.grid[1:,1:,1]
    else:
        x = sol_grid.grid[:,:,0]
        y = sol_grid.grid[:,:,1]
        # avoids ploting branch points multiple times
        if depth != 0 and not uv_lines:
            x[0,0] = np.nan
            y[0,0] = np.nan
            
        
    # cycle through default colors each sector if plt_colors=True
    if not uv_lines:
        if plt_colors:
            ax.scatter(x, y, alpha=0.5,  zorder=1, s=pt_size)
        else:
            ax.scatter(x, y, alpha=0.5, c='C0',  zorder=1, s=pt_size)
        
    if uv_lines:
        c1 = 'red'
        c2 = 'blue'
        z = 2
        if rev:
            c1 = 'blue'
            c2 = 'red'
            z=1
        lb = 0
        if plt_bds:
            lb = 1
        for i in range(1, len(x[0,:])):
            ax.plot(x[:,i],y[:,i], c=c1, zorder=z%2)
        for i in range(1, len(x[:,0])):
            ax.plot(x[i,:],y[i,:], c=c2, zorder=(z+1)%2)
        
    
    if (sol_grid.children != None):
        i = 0
        for child in sol_grid.children:
            if i % 2 != 0:
                plot_grid_pdisc(child, depth+1, fig, ax, plt_bps=plt_bps, 
                                plt_colors=plt_colors, plt_bds=plt_bds, 
                                pt_size=pt_size, uv_lines=uv_lines, rev=not rev)
            else:
                plot_grid_pdisc(child, depth+1, fig, ax, bd_sect=bd_sect, 
                                plt_bps=plt_bps, plt_colors=plt_colors, 
                                plt_bds=plt_bds, pt_size=pt_size, uv_lines=uv_lines, rev=rev)
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
        ax.plot(xbd[:,0], xbd[:,1], c='black', zorder=3)
        ax.plot(ybd[:,0], ybd[:,1], c='black', zorder=3)
        
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
            
# finds curves in asymptotic coords by scaning for first location with rho 
# value >= target starting from x-axis
def grid_scan(sol_grid, xv, yv, target):
    cx1 = []
    cy1 = []
    cx2 = []
    cy2 = []
    rho = sol_grid.grid[:,:,2]
    for i in range(sol_grid.rows):
        for j in range(sol_grid.cols):
            if rho[i,j] >= target:
                cx1.append(xv[i,j])
                cy1.append(yv[i,j])
                break
            
    for j in range(sol_grid.cols):
        for i in range(sol_grid.rows):
            if rho[i,j] >= target:
                cx2.append(xv[i,j])
                cy2.append(yv[i,j])
                break
            
    return cx1, cy1, cx2, cy2


# function to plot sectors in asymptotic coordinates  
def plot_grid_asym(sol_grid, depth=0, fig=None, ax=None, pt_size=10):
    if (sol_grid == None):
        return
    if (depth == 0):
        fig, ax = plt.subplots()
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_aspect('equal')
    
    nx, ny = (sol_grid.rows, sol_grid.cols)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    rho = sol_grid.grid[:,:,2]
    
    cx1, cy1, cx2, cy2 = grid_scan(sol_grid,xv,yv,1.8)
    cx = (np.array(cx1) + np.array(cx2[::-1])) / 2
    cy = (np.array(cy1) + np.array(cy2[::-1])) / 2
    
    
    ax.scatter(xv, yv, alpha=0.5,  zorder=1, s=pt_size)
    ax.plot(cx1,cy1, zorder=2, c='red')
    ax.plot(cx2,cy2, zorder=2, c='green')
    ax.plot(cx,cy, zorder=2, c='black')
    
    
    if depth==0:
        plt.show()
    
    if (sol_grid.children != None):
        for child in sol_grid.children:
            plot_grid_rho(child, depth+1, fig, ax)


"""
Functions for visualzing spherical Chebyshev net
"""
class Norm_plot:
    def __init__(self, norm_grid, plt_bps=False, plt_bds=False):
        self.norm_grid = norm_grid
        self.plt_bps = plt_bps
        self.plt_bds = plt_bds
        
        self.pl = pv.Plotter()
        self.plot_norm_arrows(norm_grid)
        
        if plt_bds:
            self.plot_bds(norm_grid)
        
        if plt_bps:
            self.plot_bps(norm_grid)
            
        self.pl.add_axes()
        self.pl.show()
        
    def plot_norm_arrows(self, norm_grid, depth=0):
        print(depth)
            
        c = np.array([0,0,0])
        
        lb = 0
        if self.plt_bds:
            lb = 1
        
        for i in range(lb, norm_grid.rows):
            for j in range(lb, norm_grid.cols):
                n = norm_grid.norms[i,j,:]
                if np.any(n!=0):
                    self.pl.add_arrows(c, n, mag=1, color='cyan', opacity=0.8)
        
                    
        if (norm_grid.children != None):
            for child in norm_grid.children:
                    self.plot_norm_arrows(child, depth=depth+1)
                
    def plot_bds(self, norm_grid, depth=0):
        
        c = np.array([0,0,0])
        
        for i in range(norm_grid.rows):
            n1 = norm_grid.norms[i,0,:]
            self.pl.add_arrows(c, n1, mag=1.05, color='orange')
        
        for j in range(norm_grid.cols):
            n2 = norm_grid.norms[0,j,:]
            self.pl.add_arrows(c, n2, mag=1.05, color='orange')
            
        if (norm_grid.children != None):
            for child in norm_grid.children:
                self.plot_bds(child, depth=depth+1)
                
    def plot_bps(self, norm_grid, depth=0):
        
        c = np.array([0,0,0])
        
        if (norm_grid.bp_loc != None):
            us, vs = norm_grid.bp_loc
            n = norm_grid.norms[us,vs,:]
            self.pl.add_arrows(c, n, mag=1.1, color='red')
        
        
        if (norm_grid.children != None):
            for child in norm_grid.children:
                self.plot_bps(child, depth=depth+1)
                

class Arc_plot:
    def __init__(self, norm_grid, plt_bps=False, plt_bds=False, depth_dis=False):
        self.norm_grid = norm_grid
        self.plt_bps = plt_bps
        self.plt_bds = plt_bds
        self.depth_dis = depth_dis
        
        self.pl = pv.Plotter()
        self.plot_arcs(norm_grid)
        
        if plt_bds:
            self.plot_bds(norm_grid)
        
        if plt_bps:
            self.plot_bps(norm_grid)
            
        self.pl.add_axes()
        self.pl.show()
        
        
    def plot_arcs(self, norm_grid, depth=0):
            
        c = np.array([0,0,0])
        
        lb = 0
        # if self.plt_bds:
        #     lb = 1
        for i in range(lb, norm_grid.rows-1):
            for j in range(lb, norm_grid.cols-1):
                
                scale = 1
                if self.depth_dis:
                    scale = 1 + 0.5 * depth

                n0 = scale * norm_grid.norms[i  ,   j,:]
                n1 = scale * norm_grid.norms[i  , j+1, :]
                n2 = scale * norm_grid.norms[i+1,   j, :]
                n12 = scale * norm_grid.norms[i+1, j+1, :]
                
                c1 = 'blue'
                c2 = 'red'
                
                if np.any(n0 != 0) and np.any(n1 != 0) and i==0 and not self.plt_bds:
                    u1 = pv.CircularArc(n0, n1, c, resolution=10)
                    self.pl.add_mesh(u1, line_width=5, color=c1)
                if np.any(n0 != 0) and np.any(n2 != 0) and j==0 and not self.plt_bds:
                    v1 = pv.CircularArc(n0, n2, c, resolution=10)
                    self.pl.add_mesh(v1, line_width=5, color=c2)
                if np.any(n2 != 0) and np.any(n12 != 0):
                    u2 = pv.CircularArc(n2, n12, c, resolution=10)
                    self.pl.add_mesh(u2, line_width=5, color=c1)
                if np.any(n1 != 0) and np.any(n12 != 0):
                    v2 = pv.CircularArc(n1, n12, c, resolution=10)
                    self.pl.add_mesh(v2, line_width=5, color=c2)
        
                
        if (norm_grid.children != None):
            for child in norm_grid.children:
                self.plot_arcs(child, depth=depth+1)
                
           
    def plot_bds(self, norm_grid, depth=0):
        
        c = np.array([0,0,0])
        scale = 1
        if self.depth_dis:
            scale = 1 + 0.5 * depth
        
        for i in range(norm_grid.rows - 1):
            n1 = scale * norm_grid.norms[i,0,:]
            n11 = scale * norm_grid.norms[i+1,0,:]
            if np.any(n1 != 0) and np.any(n11 != 0):
                v1 = pv.CircularArc(n1, n11, c, resolution=10)
                self.pl.add_mesh(v1, line_width=5, color="black")
            
            
        for i in range(norm_grid.cols - 1):
            n2 = scale * norm_grid.norms[0,i,:]
            n22 = scale * norm_grid.norms[0,i+1,:]
            if np.any(n2 != 0) and np.any(n22 != 0):
                u1 = pv.CircularArc(n2, n22, c, resolution=10)
                self.pl.add_mesh(u1, line_width=5, color="black")
            
        
        

        if (norm_grid.children != None):
            for child in norm_grid.children:
                self.plot_bds(child, depth=depth+1)
                
                
    def plot_bps(self, norm_grid, depth=0):
        
        if (norm_grid.bp_loc != None):
            us, vs = norm_grid.bp_loc
            print('Depth:',depth, ',',us,vs)
            n = norm_grid.norms[us,vs,:]
            
            if self.depth_dis:
                scale0 = 1 + 0.5 * depth 
                scale1 = 1 + 0.5 * (depth + 1) 
                n0 = n * scale0
                n1 = n * scale1
                self.pl.add_points(n0, render_points_as_spheres=True, point_size=20, color='red')
                self.pl.add_points(n1, render_points_as_spheres=True, point_size=20, color='red')
            else:
                self.pl.add_points(n, render_points_as_spheres=True, point_size=20, color='red')
                
        if (norm_grid.children != None):
            for child in norm_grid.children:
                self.plot_bps(child, depth=depth+1)
                    
                    
class Surf_plot:
    def __init__(self, surf_grid, plt_bps=False, plt_bds=False, depth_dis=False):
        self.grid = surf_grid.grid
        
        self.pl = pv.Plotter()
        
        self.poly_plot(surf_grid)
        
        self.pl.add_axes()
        self.pl.show()
        
    def create_pt_ar(self, surf_grid):
        
        pts = []
        for i in range(surf_grid.rows):
            for j in range(surf_grid.cols):
                r = surf_grid.grid[i,j,:]
                if not (np.any(np.isnan(r))):
                    pts.append(r)
        return pts
        
        
    def create_pt_map(self, surf_grid):
        
        pt_map = {}
        n = 0
        for i in range(surf_grid.rows):
            for j in range(surf_grid.cols):
                r = surf_grid.grid[i,j,:]
                if not (np.any(np.isnan(r))):
                    pt_map[r.tobytes()] = n
                    n+=1
                
        return pt_map
        
                
    def poly_plot(self, surf_grid, depth=0):
        
        verts = self.create_pt_ar(surf_grid)
        pt_map = self.create_pt_map(surf_grid)
        
        faces = []
        grid = surf_grid.grid 

        for i in range(surf_grid.rows-1):
            for j in range(surf_grid.cols-1):
                
                r0 =  grid[i  ,j  ,:]
                r1 =  grid[i  ,j+1,:]
                r2 =  grid[i+1,j  ,:]
                r12 = grid[i+1,j+1,:]
                
                # if depth == 0:
                #     us,vs = surf_grid.bp_loc
                #     bp = grid[us,vs,:]
                #     bp1 = grid[us+1,vs,:]
                #     cloud2 = pv.PolyData([bp, bp1])
                #     self.pl.add_mesh(cloud2, vertex_color='r', render_points_as_spheres=True, show_vertices=True,
                #         point_size=10)
                
                if i == 0 and j == 0 and depth !=0:
                    cloud2 = pv.PolyData([r0, r1, r2])
                    self.pl.add_mesh(cloud2, show_edges=True, line_width=1, vertex_color='r',
                        render_points_as_spheres=True, show_vertices=True,
                        point_size=10)
                    
                
                # if surf_grid.bp_loc != None:
                #     us,vs = surf_grid.bp_loc
                #     cloud = pv.PolyData([grid[us,vs,:], grid[us+1,vs,:]])
                #     self.pl.add_mesh(cloud, vertex_color='r', render_points_as_spheres=True, show_vertices=True,
                #         point_size=10)
                
                if not(np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(r2)) or np.any(np.isnan(r12))):
                    faces.append([4, pt_map[r0.tobytes()], pt_map[r1.tobytes()], pt_map[r12.tobytes()], pt_map[r2.tobytes()]])
                    
        surf = pv.PolyData(verts, faces)
        
        self.pl.add_mesh(surf, show_edges=True, line_width=1, vertex_color='r')
        
        if surf_grid.children != None:

                self.poly_plot(surf_grid.children[2], depth=depth+1)
                self.poly_plot(surf_grid.children[0], depth=depth+1)
                self.poly_plot(surf_grid.children[1], depth=depth+1)
            
                
            # for child in surf_grid.children:
            #     self.poly_plot(child, depth=depth+1)
            #     break
        
        
    def delaunay_plot(self, surf_grid):
        
        pts = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                pts.append(self.grid[i,j,:])
        
        cloud = pv.PolyData(pts)
        surf = cloud.delaunay_2d()
        self.pl.add_mesh(surf, show_edges=True)
                    