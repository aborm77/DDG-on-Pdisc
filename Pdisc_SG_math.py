# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:22:20 2024

@author: Ari Bormanis
Purpose: Some math functions for DDG solving of the Sine-Gordon equation on 
the Poincare disk
"""

import numpy as np

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


# Gets the approx angle for the Sin-Gordon equation
def norm(w):
    return np.sqrt(w[0]**2 + w[1]**2 )


# Computes the angle betwen two vectors using the dot product
def angle(w1,w2):
    dot = w1[0]*w2[0] + w1[1]*w2[1]
    return np.arccos(dot / (norm(w1) * norm(w2)))


# Computes geodesic distance of a point from the origin
def geo_dist(w):
    return 2*np.arctan(norm(w))


