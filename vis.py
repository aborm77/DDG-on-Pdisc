# -*- coding: utf-8 -*-
"""
Visualization for the DDG K-surface pipeline on the Poincaré disk.

Provides one plotting class per pipeline stage:

    Pdisc_plot  — Chebyshev net on the Poincaré disk (matplotlib)
    Arc_plot    — Chebyshev net on the unit sphere (PyVista)
    Surf_plot   — embedded surface in R³ (PyVista)

Author: Ari Bormanis
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def geo_circ(R):
    """Return (x, y) points on the geodesic circle of hyperbolic radius R in the first quadrant."""
    t = np.linspace(0, np.pi/2, 50)
    x = np.tanh(R / 2) * np.cos(t)
    y = np.tanh(R / 2) * np.sin(t)
    return x, y
    

class Pdisc_plot:
    """
    Plots the Chebyshev net on the Poincaré disk. Supports scatter plots of grid
    vertices (plot_points) and colored u/v-coordinate line plots (plot_lines).
    Sector boundaries and branch point locations can be overlaid via plot_bds and
    plot_bps. The ρ field can be visualized separately as a 3D surface via
    plot_grid_rho, which creates its own figure independent of __init__.
    """
    def __init__(self, sol_grid, plt_pts=False, plt_lines=True,
                 plt_bps=True, plt_colors=False, plt_bds=True,
                 save=False, f_name=None, pt_size=10, rev=False):
        self.plt_colors = plt_colors
        self.plt_bds = plt_bds
        self.pt_size = pt_size

        self.fig, self.ax = plt.subplots()
        rmax = sol_grid.rmax
        self.ax.set_xlim(-0.1, rmax + 0.14)
        self.ax.set_ylim(-0.1, rmax + 0.14)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal')
        circ_x, circ_y = geo_circ(sol_grid.R)
        self.ax.plot(circ_x, circ_y, c='green', zorder=0)

        if plt_pts:
            self.plot_points(sol_grid)
        if plt_lines:
            self.plot_lines(sol_grid, rev=rev)
        if plt_bds:
            self.plot_bds(sol_grid)
        if plt_bps:
            self.plot_bps(sol_grid)

        if save:
            fname = f_name if f_name else 'test'
            self.fig.savefig(f'figs/{fname}.png', dpi=100)
            plt.close(self.fig)
        else:
            plt.show()

    def plot_points(self, sol_grid, depth=0, bd_sect=False):
        """Scatter-plot the grid vertices, recursing over child sectors.

        bd_sect=True skips the first row and column to avoid double-plotting
        shared boundary points with the parent sector.
        """
        if sol_grid is None:
            return

        if bd_sect:
            x = sol_grid.grid[1:, 1:, 0]
            y = sol_grid.grid[1:, 1:, 1]
        else:
            x = sol_grid.grid[:, :, 0].copy()
            y = sol_grid.grid[:, :, 1].copy()
            if depth != 0:
                x[0, 0] = np.nan
                y[0, 0] = np.nan

        if self.plt_colors:
            self.ax.scatter(x, y, alpha=0.5, zorder=1, s=self.pt_size)
        else:
            self.ax.scatter(x, y, alpha=0.5, c='C0', zorder=1, s=self.pt_size)

        if sol_grid.children is not None:
            for i, child in enumerate(sol_grid.children):
                bd = bd_sect if i % 2 == 0 else False
                self.plot_points(child, depth=depth+1, bd_sect=bd)

    def plot_lines(self, sol_grid, depth=0, rev=False):
        """Plot u- and v-coordinate lines of the Chebyshev net in red and blue.

        The rev flag swaps colors and is toggled for the middle (ams) child sector,
        whose u/v axes are exchanged relative to its parent.
        """
        if sol_grid is None:
            return

        x = sol_grid.grid[:, :, 0]
        y = sol_grid.grid[:, :, 1]

        c1, c2 = ('red', 'blue') if not rev else ('blue', 'red')
        z = 2 if not rev else 1

        for i in range(1, x.shape[1]):
            self.ax.plot(x[:, i], y[:, i], c=c1, zorder=z % 2)
        for i in range(1, x.shape[0]):
            self.ax.plot(x[i, :], y[i, :], c=c2, zorder=(z + 1) % 2)

        if sol_grid.children is not None:
            for i, child in enumerate(sol_grid.children):
                child_rev = (not rev) if i % 2 != 0 else rev
                self.plot_lines(child, depth=depth+1, rev=child_rev)

    def plot_bds(self, sol_grid, depth=0):
        """Plot the sector boundary rays in black."""
        if sol_grid is None:
            return

        xbd = sol_grid.xbd[:, :2]
        ybd = sol_grid.ybd[:, :2]
        self.ax.plot(xbd[:, 0], xbd[:, 1], c='black', zorder=3)
        self.ax.plot(ybd[:, 0], ybd[:, 1], c='black', zorder=3)

        if sol_grid.children is not None:
            for child in sol_grid.children:
                self.plot_bds(child, depth=depth+1)

    def plot_bps(self, sol_grid, depth=0):
        """Plot branch point locations as red markers."""
        if sol_grid is None:
            return

        if sol_grid.bp_loc is not None:
            loc1, loc2 = sol_grid.bp_loc
            self.ax.scatter(sol_grid.grid[loc1, loc2, 0],
                            sol_grid.grid[loc1, loc2, 1],
                            c='red', zorder=4, s=self.pt_size)

        if sol_grid.children is not None:
            for child in sol_grid.children:
                self.plot_bps(child, depth=depth+1)

    def plot_grid_rho(self, sol_grid, depth=0, _ax=None):
        """Plot the ρ values as a 3D surface over the Poincaré disk."""
        if sol_grid is None:
            return
        if depth == 0:
            _, _ax = plt.subplots(subplot_kw=dict(projection='3d'))
            _ax.set_xlabel('X')
            _ax.set_ylabel('Y')
            _ax.set_zlabel('rho')

        x = sol_grid.grid[:, :, 0]
        y = sol_grid.grid[:, :, 1]
        rho = sol_grid.grid[:, :, 2]
        _ax.plot_surface(x, y, rho, alpha=0.5, color='C'+str(depth))

        if sol_grid.children is not None:
            for child in sol_grid.children:
                self.plot_grid_rho(child, depth=depth+1, _ax=_ax)

        if depth == 0:
            plt.show()
                

class Arc_plot:
    """
    Plots the Chebyshev net on the sphere. Due to the geometry of how the
    Chebyshev net on the Poincaré disk maps to the sphere, the spherical
    Chebyshev net maps over itself. To aid with visualization this code defaults
    to plotting subsequent generations of Chebyshev nets on larger spheres so
    the multiple covering of the sphere is easier to visualize.
    """
    def __init__(self, norm_grid, plt_bps=True, plt_bds=True, depth_dis=True):
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
        """Plot the circular arcs forming the sides of each spherical rhombus in the Chebyshev net.

        All arcs are accumulated into per-color buffers across the full recursive traversal,
        then merged into one PolyData per color and added to the plotter in a single
        add_mesh call each. This avoids per-arc GPU overhead from thousands of individual
        add_mesh calls.
        """
        if depth == 0:
            self._arc_buf = {'blue': [], 'red': []}

        c = np.array([0, 0, 0])
        scale = 1 + 0.5 * depth if self.depth_dis else 1

        for i in range(norm_grid.rows - 1):
            for j in range(norm_grid.cols - 1):
                n0  = scale * norm_grid.norms[i,     j, :]
                n1  = scale * norm_grid.norms[i,   j+1, :]
                n2  = scale * norm_grid.norms[i+1,   j, :]
                n12 = scale * norm_grid.norms[i+1, j+1, :]

                if np.any(n0 != 0) and np.any(n1 != 0) and i == 0 and not self.plt_bds:
                    self._arc_buf['blue'].append(pv.CircularArc(n0, n1, c, resolution=10))
                if np.any(n0 != 0) and np.any(n2 != 0) and j == 0 and not self.plt_bds:
                    self._arc_buf['red'].append(pv.CircularArc(n0, n2, c, resolution=10))
                if np.any(n2 != 0) and np.any(n12 != 0):
                    self._arc_buf['blue'].append(pv.CircularArc(n2, n12, c, resolution=10))
                if np.any(n1 != 0) and np.any(n12 != 0):
                    self._arc_buf['red'].append(pv.CircularArc(n1, n12, c, resolution=10))

        if norm_grid.children is not None:
            for child in norm_grid.children:
                self.plot_arcs(child, depth=depth+1)

        if depth == 0:
            for color, meshes in self._arc_buf.items():
                if meshes:
                    self.pl.add_mesh(pv.merge(meshes), line_width=5, color=color)
            del self._arc_buf
                
           
    def plot_bds(self, norm_grid, depth=0):
        """Plot the sector boundary arcs in black, merged into a single mesh."""
        if depth == 0:
            self._bd_buf = []

        c = np.array([0, 0, 0])
        scale = 1 + 0.5 * depth if self.depth_dis else 1

        for i in range(norm_grid.rows - 1):
            n1  = scale * norm_grid.norms[i,   0, :]
            n11 = scale * norm_grid.norms[i+1, 0, :]
            if np.any(n1 != 0) and np.any(n11 != 0):
                self._bd_buf.append(pv.CircularArc(n1, n11, c, resolution=10))

        for i in range(norm_grid.cols - 1):
            n2  = scale * norm_grid.norms[0, i,   :]
            n22 = scale * norm_grid.norms[0, i+1, :]
            if np.any(n2 != 0) and np.any(n22 != 0):
                self._bd_buf.append(pv.CircularArc(n2, n22, c, resolution=10))

        if norm_grid.children is not None:
            for child in norm_grid.children:
                self.plot_bds(child, depth=depth+1)

        if depth == 0:
            if self._bd_buf:
                self.pl.add_mesh(pv.merge(self._bd_buf), line_width=5, color='black')
            del self._bd_buf
                
                
    def plot_bps(self, norm_grid, depth=0):
        """Plot branch point locations as red sphere markers."""
        if norm_grid.bp_loc is not None:
            us, vs = norm_grid.bp_loc
            n = norm_grid.norms[us, vs, :]

            if self.depth_dis:
                scale0 = 1 + 0.5 * depth
                scale1 = 1 + 0.5 * (depth + 1)
                self.pl.add_points(n * scale0, render_points_as_spheres=True, point_size=20, color='red')
                self.pl.add_points(n * scale1, render_points_as_spheres=True, point_size=20, color='red')
            else:
                self.pl.add_points(n, render_points_as_spheres=True, point_size=20, color='red')

        if norm_grid.children is not None:
            for child in norm_grid.children:
                self.plot_bps(child, depth=depth+1)
                    
                    
class Surf_plot:
    """
    Plots the R³ surface generated by the Lelieuvre map applied to the branched
    Chebyshev net on the sphere.
    """
    def __init__(self, surf_grid, plt_bps=False, plt_bds=False, depth_dis=False):
        self.grid = surf_grid.grid

        self.pl = pv.Plotter()
        self.poly_plot(surf_grid)
        self.pl.add_axes()
        self.pl.show()

    def create_pt_ar(self, surf_grid):
        """Return a list of all non-NaN vertex positions in surf_grid."""
        pts = []
        for i in range(surf_grid.rows):
            for j in range(surf_grid.cols):
                r = surf_grid.grid[i, j, :]
                if not np.any(np.isnan(r)):
                    pts.append(r)
        return pts

    def create_pt_map(self, surf_grid):
        """Return a dict mapping each non-NaN vertex (by bytes key) to its index in create_pt_ar."""
        pt_map = {}
        n = 0
        for i in range(surf_grid.rows):
            for j in range(surf_grid.cols):
                r = surf_grid.grid[i, j, :]
                if not np.any(np.isnan(r)):
                    pt_map[r.tobytes()] = n
                    n += 1
        return pt_map

    def poly_plot(self, surf_grid, depth=0):
        """Build a pyvista PolyData mesh from surf_grid quads and add it to the plotter."""
        verts = self.create_pt_ar(surf_grid)
        pt_map = self.create_pt_map(surf_grid)

        faces = []
        grid = surf_grid.grid

        for i in range(surf_grid.rows - 1):
            for j in range(surf_grid.cols - 1):
                r0  = grid[i,   j,   :]
                r1  = grid[i,   j+1, :]
                r2  = grid[i+1, j,   :]
                r12 = grid[i+1, j+1, :]

                if i == 0 and j == 0 and depth != 0:
                    cloud2 = pv.PolyData([r0, r1, r2])
                    self.pl.add_mesh(cloud2, show_edges=True, line_width=1, vertex_color='r',
                                     render_points_as_spheres=True, show_vertices=True, point_size=10)

                if not (np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or
                        np.any(np.isnan(r2)) or np.any(np.isnan(r12))):
                    faces.append([4, pt_map[r0.tobytes()], pt_map[r1.tobytes()],
                                  pt_map[r12.tobytes()], pt_map[r2.tobytes()]])

        self.pl.add_mesh(pv.PolyData(verts, faces), show_edges=True, line_width=1, vertex_color='r')

        if surf_grid.children is not None:
            self.poly_plot(surf_grid.children[2], depth=depth+1)
            self.poly_plot(surf_grid.children[0], depth=depth+1)
            self.poly_plot(surf_grid.children[1], depth=depth+1)