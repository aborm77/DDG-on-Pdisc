# -*- coding: utf-8 -*-
"""
Main file that allows one to plot K-surfaces and their underlying Chebysehv nets
on the Poincare disk and sphere. Defaults to plotting all three.

Implements the geometric operations underlying the discrete K-surface construction
described in:

    Shearman, T. L. and Venkataramani, S. C. (2021). Distributed branch points and
    the shape of elastic surfaces with constant negative curvature. Journal of
    Nonlinear Science, 31(1), 13.

Author: Ari Bormanis
"""
import classes
import vis
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Generate branched discrete K-surfaces.')
    parser.add_argument('--phi0', '-p', type=float, default=np.pi/3)
    parser.add_argument('--cutoff', '-c', type=float, default=2.3)
    parser.add_argument('--radius', '-R', type=float, default=1.5)
    parser.add_argument('--seperation', '-s', type=float, default=0.1)
    parser.add_argument('--bp_algorithm', '-bp', type=str, default='bp1',
                        choices=['bp1','bp2'])
    
    args = parser.parse_args()
    
    # If you are viewing this in an IDE you can change the main parameters here
    phi0 = args.phi0
    cutoff = args.cutoff
    R = args.radius
    sep = args.seperation
    jeff = classes.Sol_tree(phi0, cutoff, R, sep)
    
    # Interesting params for bp2
    # phi0 = np.pi/6
    # cutoff = 3 * phi0
    # R = 2.5
    # sep = 0.1
    
    if args.bp_algorithm == 'bp1':
        jeff.bp1()
    else:
        jeff.bp2()
    
    
    vis.Pdisc_plot(jeff.base)
    
    norman = classes.Norm_tree(jeff)
    vis.Arc_plot(norman.norms_base)
   
    sherman = classes.Surf_tree(norman)
    vis.Surf_plot(sherman.base)

if __name__ == '__main__':
    main()