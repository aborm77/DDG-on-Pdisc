# -*- coding: utf-8 -*-
"""
Generates growth and rotation gifs for K-surfaces and their Chebyshev nets.

Run from the repo root:
    python numerical_explorations/gif_grow_and_rotate.py

Author: Ari Bormanis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIGS_DIR  = os.path.join(_REPO_ROOT, 'figs')
_GIFS_DIR  = os.path.join(_REPO_ROOT, 'gifs')

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import gc

import vis
import mesh as mesh_mod
from classes import Sol_tree, Norm_tree, Surf_tree
from gif_maker import make_gif


def _collect_arcs(norm_grid, buf=None, depth=0):
    """Recursively collect sphere arc PolyData into a color-keyed buffer.

    Each generation is scaled by 1 + 0.5 * depth (depth_dis), matching Arc_plot.
    """
    if buf is None:
        buf = {'blue': [], 'red': []}

    c = np.array([0, 0, 0])
    scale = 1 + 0.5 * depth
    for i in range(norm_grid.rows - 1):
        for j in range(norm_grid.cols - 1):
            n0  = scale * norm_grid.norms[i,     j, :]
            n1  = scale * norm_grid.norms[i,   j+1, :]
            n2  = scale * norm_grid.norms[i+1,   j, :]
            n12 = scale * norm_grid.norms[i+1, j+1, :]

            if np.any(n0 != 0) and np.any(n1 != 0) and i == 0:
                buf['blue'].append(pv.CircularArc(n0, n1, c, resolution=10))
            if np.any(n0 != 0) and np.any(n2 != 0) and j == 0:
                buf['red'].append(pv.CircularArc(n0, n2, c, resolution=10))
            if np.any(n2 != 0) and np.any(n12 != 0):
                buf['blue'].append(pv.CircularArc(n2, n12, c, resolution=10))
            if np.any(n1 != 0) and np.any(n12 != 0):
                buf['red'].append(pv.CircularArc(n1, n12, c, resolution=10))

    if norm_grid.children is not None:
        for child in norm_grid.children:
            _collect_arcs(child, buf=buf, depth=depth+1)

    return buf


def _collect_bds(norm_grid, buf=None, depth=0):
    """Recursively collect sphere sector boundary arcs into a buffer.

    Each generation is scaled by 1 + 0.5 * depth (depth_dis), matching Arc_plot.
    """
    if buf is None:
        buf = []

    c = np.array([0, 0, 0])
    scale = 1 + 0.5 * depth
    for i in range(norm_grid.rows - 1):
        n1  = scale * norm_grid.norms[i,   0, :]
        n11 = scale * norm_grid.norms[i+1, 0, :]
        if np.any(n1 != 0) and np.any(n11 != 0):
            buf.append(pv.CircularArc(n1, n11, c, resolution=10))

    for j in range(norm_grid.cols - 1):
        n2  = scale * norm_grid.norms[0, j,   :]
        n22 = scale * norm_grid.norms[0, j+1, :]
        if np.any(n2 != 0) and np.any(n22 != 0):
            buf.append(pv.CircularArc(n2, n22, c, resolution=10))

    if norm_grid.children is not None:
        for child in norm_grid.children:
            _collect_bds(child, buf=buf, depth=depth+1)

    return buf


def _collect_bps_sphere(norm_grid, buf=None, depth=0):
    """Recursively collect branch point positions scaled for depth_dis rendering.

    For each node with a branch point, appends positions at both the current
    generation's scale and the next, matching Arc_plot.plot_bps.
    """
    if buf is None:
        buf = []

    if norm_grid.bp_loc is not None:
        us, vs = norm_grid.bp_loc
        n = norm_grid.norms[us, vs, :]
        buf.append(n * (1 + 0.5 * depth))
        if norm_grid.children is not None:
            buf.append(n * (1 + 0.5 * (depth + 1)))

    if norm_grid.children is not None:
        for child in norm_grid.children:
            _collect_bps_sphere(child, buf=buf, depth=depth+1)

    return buf


def sweep_pdisc(R_values, out_dir, phi0=np.pi/3, cutoff=2.1, sep=0.1):
    """Save Poincaré disk frames for a sweep over R_values."""
    plt.switch_backend('agg')
    frame_dir = os.path.join(_FIGS_DIR, out_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, R in enumerate(R_values):
        jeff = Sol_tree(phi0, cutoff, R, sep)
        jeff.bp1()
        pdisc = vis.Pdisc_plot(jeff.base, save=False)
        pdisc.fig.savefig(os.path.join(frame_dir, f'fig{i}.png'), dpi=100)
        plt.close(pdisc.fig)
        del jeff
        gc.collect()


def sweep_sphere(R_values, out_dir, phi0=np.pi/4, cutoff=2.1, sep=0.1):
    """Save sphere frames for a sweep over R_values.

    Renders arcs (red/blue) and sector boundaries (black) with branch points
    (red spheres), matching the default Arc_plot view. Camera auto-zooms via
    reset_camera() as the net grows.
    """
    frame_dir = os.path.join(_FIGS_DIR, out_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, R in enumerate(R_values):
        jeff = Sol_tree(phi0, cutoff, R, sep)
        jeff.bp1()
        norman = Norm_tree(jeff)
        norm_grid = norman.norms_base

        arc_buf = _collect_arcs(norm_grid)
        bd_buf = _collect_bds(norm_grid)

        pl = pv.Plotter(off_screen=True)
        for color, arcs in arc_buf.items():
            if arcs:
                pl.add_mesh(pv.merge(arcs), line_width=5, color=color)
        if bd_buf:
            pl.add_mesh(pv.merge(bd_buf), line_width=5, color='black')

        # branch points at depth-displaced positions
        bp_pts = _collect_bps_sphere(norm_grid)
        for bp in bp_pts:
            pl.add_points(bp, render_points_as_spheres=True,
                          point_size=20, color='red')

        pl.reset_camera()
        pl.camera.Azimuth(-120)
        pl.camera.Elevation(50)
        pl.screenshot(os.path.join(frame_dir, f'fig{i}.png'))
        pl.close()

        del jeff, norman, arc_buf, bd_buf, pl
        gc.collect()


def sweep_surf(R_values, out_dir, phi0=np.pi/3, cutoff=2.1, sep=0.1):
    """Save surface frames for a sweep over R_values.

    Renders the uv-wireframe (red/blue), sector boundaries (black), and branch
    points (red spheres), matching the default Surf_plot view. Camera auto-zooms
    via reset_camera() as the surface grows.
    """
    frame_dir = os.path.join(_FIGS_DIR, out_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, R in enumerate(R_values):
        jeff = Sol_tree(phi0, cutoff, R, sep)
        jeff.bp1()
        norman = Norm_tree(jeff)
        sherman = Surf_tree(norman)
        surf = mesh_mod.Surf_create(sherman.base)

        pl = pv.Plotter(off_screen=True)

        # uv-wireframe and sector boundaries via Surf_plot methods
        sp = vis.Surf_plot.__new__(vis.Surf_plot)
        sp.surf = surf
        sp.pl = pl
        sp.surf_grid = sherman.base
        sp.plot_wireframe(sherman.base)
        sp.plot_bds(sherman.base)
        sp.plot_bps()

        pl.reset_camera()
        pl.camera.Azimuth(45)
        pl.screenshot(os.path.join(frame_dir, f'fig{i}.png'))
        pl.close()

        del jeff, norman, sherman, surf, sp, pl
        gc.collect()


def rotate_surf(R, n_frames, out_dir, phi0=np.pi/3, cutoff=2.1, sep=0.1):
    """Save frames of a counter-clockwise camera orbit around the z-axis.

    Builds the mesh once, then repositions the camera for each frame. The orbit
    is counter-clockwise when viewed from above (increasing angle in the x-y plane).
    """
    frame_dir = os.path.join(_FIGS_DIR, out_dir)
    os.makedirs(frame_dir, exist_ok=True)
    jeff = Sol_tree(phi0, cutoff, R, sep)
    jeff.bp1()
    norman = Norm_tree(jeff)
    sherman = Surf_tree(norman)
    surf = mesh_mod.Surf_create(sherman.base)

    center = np.array(surf.mesh.center)
    dist = surf.mesh.length * 1.2
    cam_z = center[2] + dist * 0.4

    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)):
        cam_pos = (
            center[0] + dist * np.cos(angle),
            center[1] + dist * np.sin(angle),
            cam_z,
        )
        pl = pv.Plotter(off_screen=True)

        pl.add_mesh(surf.mesh, show_edges=False)

        sp = vis.Surf_plot.__new__(vis.Surf_plot)
        sp.surf = surf
        sp.pl = pl
        sp.surf_grid = sherman.base
        sp.plot_bps()

        pl.camera_position = [cam_pos, tuple(center), (0, 0, 1)]
        pl.screenshot(os.path.join(frame_dir, f'fig{i}.png'))
        pl.close()

    del jeff, norman, sherman, surf
    gc.collect()


if __name__ == '__main__':
    R_values = np.round(np.linspace(1, 4.5, 36),1)

    # sweep_pdisc(R_values, 'grow_pdisc', phi0=np.pi/4, cutoff=2.4, sep=0.1)
    # make_gif(os.path.join(_FIGS_DIR, 'grow_pdisc'), 'grow_pdisc', dur=150, gif_dir=_GIFS_DIR)

    # sweep_sphere(R_values, 'grow_sphere', phi0=np.pi/4, cutoff=2.4, sep=0.1)
    # make_gif(os.path.join(_FIGS_DIR, 'grow_sphere'), 'grow_sphere', dur=150, gif_dir=_GIFS_DIR)

    # sweep_surf(R_values, 'grow_surf', phi0=np.pi/4, cutoff=2.4, sep=0.1)
    # make_gif(os.path.join(_FIGS_DIR, 'grow_surf'), 'grow_surf', dur=150, gif_dir=_GIFS_DIR)

    rotate_surf(3, 60, 'rotate_surf', phi0=np.pi/4, cutoff=2.4, sep=0.1)
    make_gif(os.path.join(_FIGS_DIR, 'rotate_surf'), 'rotate_surf', dur=120, gif_dir=_GIFS_DIR)
