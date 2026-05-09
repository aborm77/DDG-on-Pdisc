# Creating K-Surfaces with Branch Points

<p align="center">
  <img src="gifs/old/Change_angle_bps_bndry.gif" width="65%" alt="K-surface with branch points animating over varying angle parameter">
</p>

Create and visualize discrete analogues of surfaces with constant negative Gaussian curvature! The algorithms in this repo are based on the papers 

> **"Distributed Branch Points and the Shape of Elastic Surfaces with Constant Negative Curvature"**
> Toby L. Shearman and Shankar C. Venkataramani, *Journal of Nonlinear Science*, vol. 31, no. 1, p. 13, Jan 2021.
> Available: https://doi.org/10.1007/s00332-020-09657-2

and

> **"Discrete Surfaces with Constant Negative Gaussian Curvature and the Hirota Equation"**  
> Alexander Bobenko and Ulrich Pinkall, *J. Differential Geometry* 43 (1996), 527–611.
> Available: https://page.math.tu-berlin.de/~bobenko/papers/1996_Bob_Pin_K.pdf

## Smooth Theory

Why do certain strange, curly, crenelated shapes appear in nature? We hypothesize that surfaces in nature are approximately minimizing elastic energy
and model this via an energy functional

$$ \mathcal{E}_2[r] = \begin{cases}
    \int_{\Omega} (\kappa_1^2 + \kappa_2^2) \hspace{0.5em} dA \text{ if } r\in W^{2,2}, \hspace{0.5em} dr\cdot dr=g\\
    + \infty, \text{ else.}
    \end{cases} 
$$

where $r:\Omega \to \mathbb{R}^3$ is a surface, $\kappa_1$, $\kappa_2$ are the principal curvatures, and $g$ is the target metric. In particular, curly surfaces have a boundary that is exponentially increasing with their radius. Hence a natural choice for $g$ is a hyperbolic metric (i.e a metric that yields negative Gaussian curvature $K<0$). We choose to study surfaces with constant negative Gaussian curvature $K=-1$. It appears that minimizers of this functional are non-smooth ($C^{1,1}$ or even less regular)! Therefore the normal Euler-Lagrange approach is insufficient. This motivates the...

## Discrete Differential Geometric Approach

Let us call a discrete approximation of a surface with constant negative Gaussian curvature a K-surface. We will also assume all faces of a K-surface are quadrilaterals. As mentioned in the paper by Bobenko and Pinkall, a natural discretization for K-surfaces are the conditions:

1. All immediate neighbors of a vertex lie in the same plane.
2. Opposite sides of a quadrilateral have the same length.

Building a K-surface from just these two conditions turns out to be surprisingly difficult (but stay tuned for a future project). It turns out that a special parameterization (asymptotic coordinates) for the smooth version of a K-surface forms what is known as a Chebyshev net. For us this just means we are able to take a metric of the form 

$$
ds^2 = du^2 + 2 \cos \rho(u,v) du dv + dv^2
$$

which, if you squint, looks a bit like the law of cosines for a rhombus with an interior angle $\rho$ and side lengths $du$ and $dv$! Indeed, $\rho$ is the angle between coordinate lines in the smooth case. In our discrete world, we want to approximate our surface (and thus this metric) by building a Chebyshev net in a space where K=-1. The Poincaré disk model of the hyperbolic plane is one such space. It is defined on the domain 

$$
D = \{(x,y)\in \mathbb{R}^2: x^2 + y^2 <1 \}
$$

with the metric

$$
ds_P^2 = \left(\frac{4}{(1-x^2-y^2)^2} \right)(dx^2 +dy^2).
$$

To build a discrete Chebyshev net, we construct a tiling of rhombi on the Poincaré disk. Using some differential geometry magic we then map this to a Chebyshev net on the sphere. We then can use even more dark geometry secrets to derive

$$
r_u = N_u \times N \text{ and } r_v = -N_v \times N
$$

where $N$ is the normal for our surface defined by $r$. Taking a first order discretization of these formulas then allows us to construct a surface from the Chebyshev net on the sphere! Due to some clever integrability considerations this will create a K-surface. At a high level $\rho$ must satisfy the Sine-Gordon equation $\rho_{uv} = \sin \rho$ and any solution to the Sine-Gordon equation defines a surface with $K=-1$! All we have essentially done with this algorithm is create an approximation of a solution to the Sine-Gordon equation that critically satisfies discrete analogues of smooth conditions (this is the true magic of the DDG). 

## File Structure

| File | Description |
|---|---|
| `main.py` | Entry point: CLI interface for running the full pipeline |
| `classes.py` | Core implementation: all grid and tree classes for the three-stage pipeline |
| `math_functions.py` | Mathematical utilities: Möbius transformations, Rodrigues rotation, Householder reflection |
| `vis.py` | Visualization: 2D Poincaré disk plots (matplotlib) and 3D surface/sphere plots (pyvista) |
| `mesh.py` | Mesh construction and I/O: builds, saves, and loads pyvista surface meshes |
| `gif_maker.py` | Helper script for assembling figure sequences into animated GIFs |
| `numerical_explorations/` | Parameter sweep scripts for exploring solution families |
| `meshes/` | Pre-built mesh files (`.vtk`) ready to download and explore |

## Meshes

Starter meshes are included directly in the `meshes/` folder, organized into `vtk/` and `stl/` subfolders. Each mesh is labeled by the number of generations of branch points it contains. Larger meshes that exceed reasonable repository size limits are available on the [v1.0.0 release page](https://github.com/aborm77/DDG-on-Pdisc/releases/tag/v1.0.0).

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyVista](https://pyvista.org/)
- [Numba](https://numba.pydata.org/)
- [imageio](https://imageio.readthedocs.io/)

## Usage

Run the full pipeline from the command line:

```bash
python main.py [options]
```

By default this shows the Poincaré disk plot and the K-surface in R³ as a wireframe of asymptotic coordinate lines (red and blue). Pass `--arc` to also show the intermediate spherical arc plot.

### Geometry / solver

| Argument | Short | Default | Description |
|---|---|---|---|
| `--phi0` | `-p` | `π/3` | Initial angle parameter |
| `--cutoff` | `-c` | `2.1` | Cutoff radius for branch point placement |
| `--radius` | `-R` | `3` | Radius of the hyperbolic disk |
| `--separation` | `-s` | `0.1` | Side length of all hyperbolic rhombi |
| `--bp_algorithm` | `-bp` | `bp1` | Branch point algorithm (`bp1` or `bp2`) |

### Plots to show

| Argument | Description |
|---|---|
| `--arc` | Also show the Chebyshev net on the sphere (PyVista) |
| `--surf` | Also show the embedded K-surface in R³ (PyVista) |

### Poincaré disk plot options

| Argument | Default | Description |
|---|---|---|
| `--plt_pts` | off | Scatter-plot grid vertices |
| `--no_lines` | off | Suppress u/v-coordinate line plot |
| `--no_bps` | off | Suppress branch point markers |
| `--no_bds` | off | Suppress sector boundary lines |
| `--plt_colors` | off | Color vertices by sector |
| `--save` | off | Save figure to `figs/` instead of displaying |
| `--f_name` | `test` | Filename stem when `--save` is used |
| `--pt_size` | `10` | Marker size for scatter plots |
| `--rev` | off | Swap u/v-line colors |

### Arc plot options (used with `--arc`)

| Argument | Default | Description |
|---|---|---|
| `--no_depth_dis` | off | Render all sphere generations at the same scale instead of offset by depth |

### Surface plot options

| Argument | Default | Description |
|---|---|---|
| `--plt_surf_mesh` | off | Overlay the solid quad surface mesh |
| `--no_wireframe` | off | Suppress the asymptotic coordinate wireframe |
| `--no_bps` | off | Suppress branch point markers on the surface |
| `--no_bds` | off | Suppress sector boundary lines on the surface |
| `--save_surf` | off | Save the mesh to `meshes/` |
| `--f_name_surf` | `test` | Filename stem when `--save_surf` is used |
| `--f_type_surf` | `vtk` | File format: `vtk`, `vtp`, `stl`, `obj` |
| `--load_surf` | — | Path to a saved mesh file; skips the pipeline and plots directly |

### Examples

```bash
# Default: Poincaré disk + asymptotic coordinate wireframe on the surface
python main.py

# Solid surface mesh only (no wireframe)
python main.py --plt_surf_mesh --no_wireframe

# Solid mesh with wireframe overlaid
python main.py --plt_surf_mesh

# Custom geometry with the arc plot
python main.py -p 0.5 -R 2.5 --arc

# Save the surface mesh to meshes/my_surface.vtk
python main.py --save_surf --f_name_surf my_surface

# Load and plot a previously saved mesh (no pipeline needed)
python main.py --load_surf meshes/my_surface.vtk

# Save the disk plot to figs/my_surface.png
python main.py --save --f_name my_surface
```

## Citation

If you use this code, please cite the paper it is based on and credit the repository:

**Paper:**
```bibtex
@Article{Shearman2021,
    author={Shearman, Toby L.
    and Venkataramani, Shankar C.},
    title={{D}istributed {B}ranch {P}oints and the {S}hape of {E}lastic {S}urfaces with {C}onstant {N}egative {C}urvature},
    journal={Journal of Nonlinear Science},
    year={2021},
    month={Jan},
    day={07},
    volume={31},
    number={1},
    pages={13},
    issn={1432-1467},
    doi={10.1007/s00332-020-09657-2},
    url={https://doi.org/10.1007/s00332-020-09657-2}
}
```

**Code:**
```bibtex
@misc{bormanis2022ksurfaces,
  author={Bormanis, Ari},
  title={DDG on Poincaré Disk: Discrete K-Surface Construction with Branch Points},
  year={2022},
  url={https://github.com/aborm77/DDG-on-Pdisc}
}
```

## AI Disclaimer 

This code predates the current age of AI coding agents! All algorithms and base logic were developed by me independent of AI coding tools. That said, I now use Claude Code to help clean up conventions, fix docstrings, and implement speedups to existing methods. Please do not train AI on this code without my express permission. 

## Summary

Make cool shapes and rock out! 
