# -*- coding: utf-8 -*-
"""
@author: Ari Bormanis
Purpose: Some math functions for DDG solving of the Sine-Gordon equation on 
the Poincare disk.
"""

import numpy as np
from numpy.linalg import norm

# householder reflection of a vector v through a plane with normal n
def house(v, n):
    return v - (2 * np.dot(v,n)) * n

# Rodrigues formula for rotation of a vector, v, by ,a, radians, around a normal, n
# rotates according to right hand rule
def rod(v, n, a):
    return (np.cos(a) * v) + (np.sin(a) * np.cross(n,v)) + (n *(np.dot(n,v)) * (1 - np.cos(a)))

# Returns 2x2 rotation matrix using angle phi
def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])

# Applies the mobius function described in the distribtued branch points paper
# The function has been rationalized so we can just apply it to real vectors
def f(pt, z0):
    dot = 1 + pt[0] * z0[0] + pt[1] * z0[1]
    cross = pt[1] * z0[0] - pt[0] * z0[1]
    denom = dot**2 + cross**2
    num = pt + z0
    x = dot * num[0] + cross * num[1]
    y = dot * num[1] - cross * num[0]
    return np.array([x,y]) / denom


# Computes the last vertex in the quadralaterial like in the branch points paper
def w12_comp(w1, w2):
    num = w1 + w2
    denom = 1 + np.sqrt((w1[0]*w2[0] - w1[1]*w2[1])**2 + (w1[0]*w2[1] + w1[1]*w2[0])**2)
    return num / denom

# Computes the angle betwen two vectors using the dot product
def angle(w1,w2):
    return np.arccos(np.dot(w1,w2) / (norm(w1) * norm(w2)))

# Computes geodesic distance of a point from the origin in the Poincare disk
def geo_dist(w):
    return 2*np.arctanh(norm(w))


