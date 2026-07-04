# -*- coding: utf-8 -*-
"""
Numerical experiments for branched surfaces.

Author: Ari Bormanis
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

import vis
from classes import Sol_tree
import math_functions as mf



# An experiment for reasoning about the complexity / running time
def prune_and_count(sol_grid, R):
    """NaN-mask every grid point whose geodesic distance from the origin is >= R
    and counts each quad face.
    """
    count = 0
    for i in range(sol_grid.rows):
        for j in range(sol_grid.cols):
            if i != sol_grid.rows - 1 and j != sol_grid.cols-1 and mf.geo_dist(sol_grid.grid[i, j, :2]) <= R and mf.geo_dist(sol_grid.grid[i+1, j+1, :2]) <= R and mf.geo_dist(sol_grid.grid[i+1, j, :2]) <= R and mf.geo_dist(sol_grid.grid[i, j+1, :2]) <= R:   
                count += 1
                
    if sol_grid.children is not None:
        child_count = prune_and_count(sol_grid.children[0], R) + prune_and_count(sol_grid.children[1], R) + prune_and_count(sol_grid.children[2], R)
        return count + child_count 
    return count 


# fixed parameters
# ns = [3, 4, 5, 6]
# cutoff = 2.1
# sep = 0.03

# data creation
# data = {}
# Rs = np.linspace(2,6.25,18)
# data['R'] = Rs
# for n in ns:
#     phi0 = np.pi / n
#     data['pi/'+str(n)] = []
#     print('n:', n)
#     for R in Rs:
#         print()
#         print(R)
#         jeff = Sol_tree(phi0, cutoff, R, sep)
#         R += 0.25
#         jeff.bp1()
#         data['pi/'+str(n)].append(2*n * prune_and_count(jeff.base, R))
        
#         del jeff
#         print()
#     gc.collect()
    
# df = pd.DataFrame(data)
# df.to_csv('data/number_of_quads.csv', index=False)

# data plotting
# ns = [3, 4, 5, 6]
# df = pd.read_csv('data/number_of_quads.csv')
# ns.reverse()
# for n in ns:
#     phi0 = 'pi/' + str(n)
#     plt.plot(df['R'][:17], np.log(df[phi0][:17]), label=r'$\pi$' + '/' + str(n))
# plt.legend()
# plt.xlabel('R')
# plt.ylabel(r'$\ln$(Number of Quads)')
# plt.rcParams.update({'font.size': 20})
# plt.savefig('data/number_of_quads.pdf', bbox_inches='tight')
# plt.show()


# An experiment for checking to see if changing the seperation effects the energetics
def prune_and_energy(sol_grid, R):
    """NaN-mask every grid point whose geodesic distance from the origin is >= R
    and counts each quad face.
    """
    count = 0
    for i in range(sol_grid.rows):
        for j in range(sol_grid.cols):
            if i != sol_grid.rows - 1 and j != sol_grid.cols-1 and mf.geo_dist(sol_grid.grid[i, j, :2]) <= R and mf.geo_dist(sol_grid.grid[i+1, j+1, :2]) <= R and mf.geo_dist(sol_grid.grid[i+1, j, :2]) <= R and mf.geo_dist(sol_grid.grid[i, j+1, :2]) <= R:   
                count += sol_grid.grid[i,j,3]
                
    if sol_grid.children is not None:
        child_count = prune_and_energy(sol_grid.children[0], R) + prune_and_energy(sol_grid.children[1], R) + prune_and_energy(sol_grid.children[2], R)
        return count + child_count 
    return count 


# Data creation
# cutoff = 1.9
# R = 2.5
# phi0 = np.pi/4

# data = {}
# seps = np.linspace(0.01,0.1,200)
# data['s_large'] = seps
# data['energy_large'] = []
# for s in seps:
#     print('s:', s)
#     jeff = Sol_tree(phi0, cutoff, R, s, energy=True)
#     jeff.bp1()
#     data['energy_large'].append(2 * 4 * prune_and_energy(jeff.base, R))
    
#     del jeff
#     print()
# gc.collect()

# plt.plot(seps, data['energy_large'])

# seps = np.linspace(0.001,0.01,50)
# data['s_small'] = seps
# data['energy_small'] = []
# for s in seps:
#     print('s:', s)
#     jeff = Sol_tree(phi0, cutoff, R, s, energy=True)
#     jeff.bp1()
#     data['energy_small'].append(2 * 4 * prune_and_energy(jeff.base, R))
    
#     del jeff
#     print()
# gc.collect()
    

# plt.plot(seps, data['energy_small'])

# df = pd.DataFrame(data)
# df.to_csv('data/sep_and_energy.csv', index=False)

# data plotting
# plt.figure(1)
# df = pd.read_csv('data/sep_and_energy.csv')
# plt.plot(df['s_large'], df['energy_large'])
# plt.xlabel('Edge Length s')
# plt.ylabel('Total Willmore Energy')
# plt.rcParams.update({'font.size': 20})

# plt.savefig('data/large_sep_en.pdf', bbox_inches='tight')
# plt.show()

# plt.figure(2)
# df = pd.read_csv('data/sep_and_energy.csv')
# plt.plot(df['s_small'], df['energy_small'])
# plt.xlabel('Edge Length s')
# plt.ylabel('Total Willmore Energy')
# plt.rcParams.update({'font.size': 20})

# plt.savefig('data/small_sep_en.pdf', bbox_inches='tight')
# plt.show()


# Experimenting to show that there is a notion of optimal phi0 and how it is ind of s
# cutoff = 1.9
# Rs = [2, 3, 4]
# ns = [2,3,4,5,6,7,8,9,10]
# seps = np.linspace(0.01,0.05,50)

# for R in Rs:
#     data = {}
#     data['s'] = seps
#     print('R:', R)
#     i = 0
#     for n in ns:
#         data['pi/' + str(n)] = []
#         phi0 = np.pi / n
#         print('n:', n)
#         for s in seps:
#             print('s:', s)
#             jeff = Sol_tree(phi0, cutoff, R, s, energy=True)
#             jeff.bp1()
#             data['pi/' + str(n)].append(2 * n * prune_and_energy(jeff.base, R))
#             del jeff
            
#             print()
#         gc.collect()
        
#     df = pd.DataFrame(data)
#     df.to_csv('data/sep_and_en_rad'+str(R)+'.csv', index=False)


# plotting data
# Rs = [2, 3, 4]
# ns = [2,3,4,5,6,7,8,9]
# cmap = plt.get_cmap('cool')
# colors = cmap(np.linspace(0.0, 1.0, len(ns)))
# for R in Rs:
#     Rdata = pd.read_csv('data/sep_and_en_rad'+str(R)+'.csv')
#     plt.figure(R)
#     i = 0
#     for n in ns:
#         plt.plot(Rdata['s'], Rdata['pi/' + str(n)], color=colors[i], label=r'$\pi$' + '/' + str(n))
#         i += 1
#     plt.title('R = '+str(R))
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', reverse=True)
#     plt.rcParams.update({'font.size': 20})
#     plt.xlabel('Edge Length s')
#     plt.ylabel('Total Willmore Energy')
#     plt.savefig('data/sep_an_en'+str(R)+'.pdf',bbox_inches='tight')
#     plt.show()


# Experimenting to see if our branched surfaces are statisfying e^{\sqrt{R}} scaling
cutoff = 1.9
n = 2
phi0 = np.pi/n
sep = 0.001
Rs= np.linspace(1,3,20)
data = {}
data['R'] = Rs
data['energy'] = []
data['n'] = []
data['const_pi/2'] = []
for R in Rs:
    print(R)
    if R >= 2:
        n = 3
    # const for comparison
    phi0 = np.pi/2
    jeff0 = Sol_tree(phi0, cutoff, R, sep, energy=True)
    jeff0.bp1()
    en0 = 2 * 2 * prune_and_energy(jeff0.base, R)
    data['const_pi/2'].append(en0)
    
    # first canidate
    print('can1')
    phi1 = np.pi / n
    jeff1 = Sol_tree(phi1, cutoff, R, sep, energy=True)
    jeff1.bp1()
    en1 = 2 * n * prune_and_energy(jeff1.base, R)
    
    data['energy'].append(en1)
    data['n'].append(n)
    
    print()
    
    del jeff0
    del jeff1
    
gc.collect()

df = pd.DataFrame(data)
df.to_csv('test_small.csv', index=False)
    
data = pd.read_csv('test.csv')
plt.plot(data['R'], np.log(data['energy']))
plt.show()