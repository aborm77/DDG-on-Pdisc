# Creating K-Surfaces with Branch Points

Create and visualize discrete analouges of surfaces with constant negative Gaussian curvature! The algorithms in this repo are based on the papers 

> **"Discrete Surfaces with Constant Negative Gaussian Curvature and the Hirota Equation"**  
> Alexander Bobenko and Ulrich Pinkall, *J. Differential Geometry* 43 (1996), 527–611.

and

> **“Distributed Branch Points and the Shape of Elastic Surfaces with Constant Negative Curvature”
> Toby L. Shearman and Shankar C. Venkataramani, *Journal of Nonlinear Science*, vol. 31, no. 1, p. 13, Jan 2021.
> Available: https://doi.org/10.1007/s00332-020-09657-2

## Theory

Why do certain strange, cruly, crenelated shapes appear in nature? We hypothesize that surfaces in nature are approximately minimizing elastic engergy
and model this via an elastic energy functional

$$ E[y] = \int_{\Omega} \kappa_1^2 + \kappa_2^2 dA .$$

where $y:\Omega \to \mathbb{R}^3$ is a surface, $\kappa_1$, $\kappa_2$ are the principle curvatures, and $g$ is the target metric. It appears that minimizers of this
functional are non-smooth ($C^{1,1}$ or even less regular)! Therefore the normal Euler-Lagrange approach is insufficent.  