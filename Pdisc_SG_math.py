# -*- coding: utf-8 -*-
"""
Math utilities for DDG solving of the Sine-Gordon equation on the Poincaré disk.

Implements the geometric operations underlying the discrete K-surface construction
described in:

    Shearman, T. L. and Venkataramani, S. C. (2021). Distributed branch points and
    the shape of elastic surfaces with constant negative curvature. Journal of
    Nonlinear Science, 31(1), 13.

Author: Ari Bormanis
"""

import numpy as np
from numpy.linalg import norm


def house(v, n):
    """Reflect vector v through the plane with unit normal n (Householder reflection)."""
    return v - (2 * np.dot(v, n)) * n


def rod(v, n, a):
    """Rotate vector v by a radians around unit axis n (Rodrigues' rotation formula).

    Rotation follows the right-hand rule.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Vector to rotate.
    n : ndarray, shape (3,)
        Unit rotation axis.
    a : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (3,)
        Rotated vector.
    """
    c, s = np.cos(a), np.sin(a)
    return c * v + s * np.cross(n, v) + (1 - c) * np.dot(n, v) * n


def rot(phi):
    """Return the 2x2 rotation matrix for angle phi (radians)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def f(pt, z0):
    """Apply a Möbius transformation of the Poincaré disk centered at z0.

    Implements the map f_{z0}(pt) = (pt + z0) / (1 + conj(z0) * pt), which is
    an isometry of the Poincaré disk that sends z0 to the origin.

    Parameters
    ----------
    pt : ndarray, shape (2,) or (N, 2)
        Point(s) in the Poincaré disk to transform.
    z0 : ndarray, shape (2,)
        Center of the transformation; must satisfy norm(z0) < 1.

    Returns
    -------
    ndarray, shape (2,) or (N, 2)
        Transformed point(s), matching the shape of pt.
    """
    p = pt[..., 0] + 1j * pt[..., 1]
    z = complex(z0[0], z0[1])
    result = (p + z) / (1 + z.conjugate() * p)
    out = np.empty_like(pt, dtype=float)
    out[..., 0] = result.real
    out[..., 1] = result.imag
    return out


def w12_comp(w1, w2):
    """Compute the fourth vertex of a Chebyshev quadrilateral given two adjacent vertices.

    Returns the unique point w12 such that (w0=origin, w1, w2, w12) forms a
    rhombus in the Poincaré disk metric, as given in Shearman & Venkataramani (2021).
    The denominator uses the identity norm(w1 * w2) = norm(w1) * norm(w2) to
    avoid an explicit complex product.

    Parameters
    ----------
    w1 : ndarray, shape (2,)
        First adjacent vertex (in Möbius-translated coordinates with w0 at origin).
    w2 : ndarray, shape (2,)
        Second adjacent vertex.

    Returns
    -------
    ndarray, shape (2,)
        Fourth vertex of the quadrilateral.
    """
    return (w1 + w2) / (1 + norm(w1) * norm(w2))


def angle(w1, w2):
    """Return the angle in radians between vectors w1 and w2."""
    return np.arccos(np.dot(w1, w2) / (norm(w1) * norm(w2)))


def geo_dist(w):
    """Return the geodesic distance from the origin to w in the Poincaré disk."""
    return 2 * np.arctanh(norm(w))
