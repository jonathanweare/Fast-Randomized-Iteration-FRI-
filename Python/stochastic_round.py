#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:40:08 2020

@author: Rob1
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:13:28 2020

@author: Rob
"""

import numpy as np
import matplotlib.pyplot as plt
import os
# os.chdir('/Users/Rob1/Desktop')
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2

#%%

#==============================
# Define the random compression
# =============================

# helper function -- marches through a list of probabilities and
# rounds them to 0 or 1
def pivotal(x):
    y = np.copy(x)
    i = 0
    j = 1
    k = 2
    d = y.size
    a = y[i]
    b = y[j]
    while k < d:
        if k < d and (a < 1e-12 or a > 1 - 1e-12):
            a = y[k]
            i = k
            k += 1
        if k < d and (b < 1e-12 or b > 1 - 1e-12):
            b = y[k]
            j = k
            k += 1
        u = np.random.rand()
        add = a + b
        if (add > 1) and (add < 2):
            if u < (1 - b)/(2 - add):
                b = add - 1
                a = 1
            else:
                a = add - 1
                b = 1
        elif (add > 0) and (add <= 1):
            if u < b / add:
                b = add
                a = 0
            else:
                a = add
                b = 0
        y[i] = a
        y[j] = b
    return(y)

# helper function -- marches through a list of probabilities and
# rounds them to 0 or 1
def systematic(x):
    y = np.copy(x)
    d = y.size
    i = -1
    mass = 0
    u = np.random.rand()
    while i < d - 1:
        i += 1
        mass += y[i]
        if mass > u:
            y[i] = 1
            mass -= 1
        else:
            y[i] = 0
    return(y)


def rows(A, x, m, method = 'stochastic', resampling = 'pivotal'):

    # Compute y = Ax    
    y = np.dot(A, x)
    compressed = np.zeros(y.size)

    # Step 1: preserve the largest entries exactly.
    # Usually terminates after a few loops.
    num_exact = 0    
    norm = np.sum(np.abs(y))
    new = np.abs(y) >= norm / m
    add = np.sum(new)
    while add > 0:
        # incorporate new preservations
        num_exact += add
        add = 0
        compressed[new] = y[new]
        y[new] = 0
        norm = np.sum(np.abs(y))
        if num_exact < m and norm > 0:
            # check for new preservations
            new = np.abs(y) >= norm / (m - num_exact)
            add = np.sum(new)

    # print(num_exact)
            
    if norm > 0:
        # Step 2: row-wise rounding
        nonzero_y = np.arange(y.size)[y != 0]
        nonzero_x = np.arange(x.size)[x != 0]
        Y = A[nonzero_y[:, None], nonzero_x] * x[nonzero_x][None, :]
        for i in range(nonzero_y.size):
            row = Y[i, :]
            # annihilation
            pos = np.sum(row[row > 0])
            neg = np.sum(row[row < 0])
            total = pos + neg
            if total > 0:
                row[row < 0] = 0
                row[row > 0] *= total/pos
            else:
                row[row > 0] = 0
                row[row < 0] *= total/neg
            # stochastic rounding
            if method == 'stochastic':
                Y[i, :] = np.random.multinomial(1, row/np.sum(row))*total
            elif method == 'absolute':
                index = np.argmax(np.abs(row))
                Y[i, :] = 0
                Y[i, index] = total
            elif method == 'relative':
                index = np.argmax(np.abs(row / x[nonzero_x]))
                Y[i, :] = 0
                Y[i, index] = total                
            
        # Step 3: assign weights to the columns
        mass = np.abs(Y) * (m - num_exact) / norm
        scaled_sums = mass.sum(axis = 0)
        floor = np.floor(scaled_sums)
        remainder = scaled_sums - floor
        if resampling == 'pivotal':
            num = floor + pivotal(remainder)
        elif resampling == 'systematic':
            num = floor + systematic(remainder)
        num = np.rint(num).astype('int')

        # print mass
        # print scaled_sums
        # print floor
        # print num
        
        # Step 4: select entries from each column
        for j in np.arange(nonzero_x.size)[num > 0]:
            # Define a lot of stuff
            f = floor[j]
            r = remainder[j]
            n = num[j]
            p = mass[:, j]
            # p_floor = np.floor(p)
            # p = p - p_floor
            # compressed[nonzero_y] = compressed[nonzero_y] + p_floor*np.sign(y)[nonzero_y] * norm / (m - num_exact)
            if r > 0:
                q_f = np.copy(p)
                q_c = np.copy(p)
                # Fix the probabilities
                for i in np.arange(p.size)[p > 0]:
                    if p[i] < r:
                        q_f[i] = 0
                        q_c[i] = p[i]/r
                    else:
                        q_f[i] = (p[i] - r)/(1 - r)
                        q_c[i] = 1
                    if np.sum(q_f) < f:
                        break
                q_f[i] += floor[j] - np.sum(q_f)
                q_c[i] += floor[j] + 1 - np.sum(q_c)
                if n == f:
                    p = q_f
                else:
                    p = q_c
            # pivotal resampling
            if resampling == 'pivotal':
                selector = nonzero_y[np.rint(pivotal(p)).astype('bool')]
            elif resampling == 'systematic':
                selector = nonzero_y[np.rint(systematic(p)).astype('bool')]
            compressed[selector] = np.sign(y)[selector] * norm / (m - num_exact)
            # compressed[selector] = compressed[selector] + np.sign(y)[selector] * norm / (m - num_exact)
    return(compressed)

def old_compress(x, m, resampling = 'pivotal'):

    compressed = np.zeros(x.size)

    # Step 1: preserve the largest entries exactly.
    # Usually terminates after a few loops.
    num_exact = 0    
    norm = np.sum(np.abs(x))
    new = np.abs(x) >= norm / m
    add = np.sum(new)
    while add > 0:
        # incorporate new preservations
        num_exact += add
        add = 0
        compressed[new] = x[new]
        x[new] = 0
        norm = np.sum(np.abs(x))
        if num_exact < m and norm > 0:
            # check for new preservations
            new = np.abs(x) >= norm / (m - num_exact)
            add = np.sum(new)
            
    if norm > 0:
        # Step 2: pivotal resampling
        mass = np.abs(x) * (m - num_exact)/norm
        if resampling == 'pivotal':
            select = np.rint(pivotal(mass)).astype('bool')
        elif resampling == 'systematic':
            select = np.rint(systematic(mass)).astype('bool')
        compressed[select] = np.sign(x)[select]*norm/(m - num_exact)

    return(compressed)

#%%

#=======================================
# Ex: 2-d diffusion with abs. boundaries
# ======================================

# Define the problem
d = 400
T = 160000
m = 16
A = np.diag(.5*np.ones(d-1), -1) + np.diag(.5*np.ones(d-1), 1)
A[:,0] = np.zeros(d)
A[0,0] = 1
A[:,d-1] = np.zeros(d)
A[d-1,d-1] = 1

x0 = np.zeros(d);
x0[np.int(d/2)-1] = 0.5
x0[np.int(d/2)] = 0.5
# print A

# A = .5*np.kron(np.eye(d), A) + .5*np.kron(A, np.eye(d))
# for j in range(np.square(d)):
#     if np.sum(A[:, j]) < 1:
#         A[:, j] = 0
#         A[j,j] = 1
# x0 = np.zeros(np.square(d))
# x0[np.int(d*(d+1)/2)] = 1.

# # Monte Carlo scheme
# mc = np.zeros(d**2)
# for i in range(m):
#     print(i)
#     i = np.int(d*(d+1)/2)
#     for t in range(T):
#         i = np.random.choice(range(d**2), p = A[:,i])
#     mc[i] += 1
# mc /= m

# # Old FRI compression
# fri = np.copy(x0)
# for t in range(T):
#     print(t)
#     fri = np.dot(A, fri)
#     fri = old_compress(fri, m)
        
# # Row-wise compression
# rowwise = np.copy(x0)
# for t in range(T):
#     print(t)
#     rowwise = rows(A, rowwise, m, resampling = 'systematic')

# Row-wise compression
absolute = np.copy(x0)
for t in range(T):
    # print("t: ",t)
    absolute = rows(A, absolute, m, method = 'absolute', resampling = 'pivotal')

print absolute
print np.sum(absolute)

# # Row-wise compression
# relative = np.copy(x0)
# for t in range(T):
#     print(t)
#     relative = rows(A, relative, m, method = 'relative', resampling = 'systematic')


#%%

# truth = np.copy(x0)
# for t in range(T):
#     if t % 100 == 0:
#         print(t)
#     truth = np.dot(A, truth)
# truth = np.sum(truth[:d])

# results = []
# for i in range(100):
#     print(i)
#     fri = np.copy(x0)
#     for t in range(T):
#         if t % 100 == 0:
#             print(t)
#         fri = np.dot(A, fri)
#         fri = old_compress(fri, m, resampling = 'systematic')
#     # rowwise = np.copy(x0)
#     # for t in range(T):
#     #     if t % 100 == 0:
#     #         print(t)
#     #     rowwise = rows(A, rowwise, m, resampling = 'systematic')
#     # results.append(np.sum(rowwise[:d]))
#     results.append(np.sum(fri[:d]))
#     # print(results[-1])
#     # print('FRI variance: ', np.mean(np.square(results - truth)))
#     # print('IID variance: ', truth*(1-truth)/100)
# plt.hist(results, bins = 25)


# #%%

# #=========
# # Plotting
# # ========

# # Compare results
# fig, ax = plt.subplots(1, 5, figsize = (20, 5))
# mc[np.int(d*(d+1)/2)] = .05
# pos0 = ax[0].imshow(mc.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[0].tick_params(width = 2)
# ax[0].set_title('Independent walkers', font)
# fri[np.int(d*(d+1)/2)] = .05
# pos1 = ax[1].imshow(fri.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[1].tick_params(width = 2)
# ax[1].set_title('Old FRI', font)
# rowwise[np.int(d*(d+1)/2)] = .05
# pos2 = ax[2].imshow(rowwise.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[2].tick_params(width = 2)
# ax[2].set_title('Random FRI', font)
# pos3 = ax[3].imshow(absolute.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[3].tick_params(width = 2)
# ax[3].set_title('Absolute FRI', font)
# pos4 = ax[4].imshow(relative.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[4].tick_params(width = 2)
# ax[4].set_title('Relative FRI', font)
# fig.tight_layout()
# cbar = fig.colorbar(pos0, ax = ax, orientation = 'vertical', fraction = .0075, pad = .02)
# fig.savefig('pivotal.png', bbox_inches = 'tight', dpi = 200)

# #%%

# #=========
# # Plotting
# # ========

# # Compare results
# fig, ax = plt.subplots(1, 5, figsize = (20, 5))
# mc[np.int(d*(d+1)/2)] = .05
# pos0 = ax[0].imshow(mc.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[0].tick_params(width = 2)
# ax[0].set_title('Independent walkers', font)
# fri[np.int(d*(d+1)/2)] = .05
# pos1 = ax[1].imshow(fri.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[1].tick_params(width = 2)
# ax[1].set_title('Old FRI', font)
# rowwise[np.int(d*(d+1)/2)] = .05
# pos2 = ax[2].imshow(rowwise.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[2].tick_params(width = 2)
# ax[2].set_title('Random FRI', font)
# pos3 = ax[3].imshow(absolute.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[3].tick_params(width = 2)
# ax[3].set_title('Absolute FRI', font)
# pos4 = ax[4].imshow(relative.reshape((d, d)), vmin = 0, vmax = .025, cmap='hot')
# ax[4].tick_params(width = 2)
# ax[4].set_title('Relative FRI', font)
# fig.tight_layout()
# cbar = fig.colorbar(pos0, ax = ax, orientation = 'vertical', fraction = .0075, pad = .02)
# fig.savefig('systematic.png', bbox_inches = 'tight', dpi = 200)