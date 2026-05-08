# -*- coding: utf-8 -*-
"""
Main file that allows one to plot K-surfaces and their underlying Chebyshev nets
on the Poincaré disk and sphere. Defaults to plotting the Poincaré disk only.

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

    # Geometry / solver
    parser.add_argument('--phi0', '-p', type=float, default=np.pi/3)
    parser.add_argument('--cutoff', '-c', type=float, default=2.1)
    parser.add_argument('--radius', '-R', type=float, default=3.2)
    parser.add_argument('--separation', '-s', type=float, default=0.1)
    parser.add_argument('--bp_algorithm', '-bp', type=str, default='bp1',
                        choices=['bp1', 'bp2'])

    # Which plots to show (Pdisc is always shown)
    parser.add_argument('--arc', action='store_true',
                        help='also show the spherical arc plot')
    parser.add_argument('--surf', action='store_true',
                        help='also show the R³ surface plot')

    # Pdisc_plot options
    parser.add_argument('--plt_pts', action='store_true')
    parser.add_argument('--no_lines', dest='plt_lines', action='store_false')
    parser.add_argument('--no_bps', dest='plt_bps', action='store_false')
    parser.add_argument('--plt_colors', action='store_true')
    parser.add_argument('--no_bds', dest='plt_bds', action='store_false')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--f_name', type=str, default=None)
    parser.add_argument('--pt_size', type=int, default=10)
    parser.add_argument('--rev', action='store_true')
    parser.set_defaults(plt_lines=True, plt_bps=True, plt_bds=True)

    # Arc_plot options
    parser.add_argument('--no_depth_dis', dest='depth_dis', action='store_false')
    parser.set_defaults(depth_dis=True)

    args = parser.parse_args()

    # If you are viewing this in an IDE you can change the main parameters here
    phi0 = args.phi0
    cutoff = args.cutoff
    R = args.radius
    sep = args.separation
    jeff = classes.Sol_tree(phi0, cutoff, R, sep)
    
    args.surf = True

    # Interesting params for bp2
    # phi0 = np.pi/6
    # cutoff = 3 * phi0
    # R = 2.5
    # sep = 0.1

    if args.bp_algorithm == 'bp1':
        jeff.bp1()
    else:
        jeff.bp2()

    vis.Pdisc_plot(jeff.base,
                   plt_pts=args.plt_pts, plt_lines=args.plt_lines,
                   plt_bps=args.plt_bps, plt_colors=args.plt_colors,
                   plt_bds=args.plt_bds, save=args.save,
                   f_name=args.f_name, pt_size=args.pt_size, rev=args.rev)

    if args.arc or args.surf:
        norman = classes.Norm_tree(jeff)
        if args.arc:
            vis.Arc_plot(norman.norms_base,
                         plt_bps=args.plt_bps, plt_bds=args.plt_bds,
                         depth_dis=args.depth_dis)
        if args.surf:
            sherman = classes.Surf_tree(norman)
            vis.Surf_plot(sherman.base)

if __name__ == '__main__':
    main()
