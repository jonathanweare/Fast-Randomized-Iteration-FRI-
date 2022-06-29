# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:13:28 2020

@author: Rob
"""

import numpy as np
import matplotlib.pyplot as plt
# import os
# os.chdir("C:\\Users\\Rob\\Desktop")
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2

#%%

#==============================
# Define the random compression
# =============================

def compression(A, x, m, proportional = True):

    # Compute y = Ax    
    y = np.dot(A, x)
    d = y.size
    compressed = np.zeros(d)

    # Step 1: preserve the largest entries exactly.
    # Usually terminates after a few loops.
    num_S = 0    
    go =  True
    while go:
        go = False
        norm = np.sum(np.abs(y))
        new = np.abs(y) > norm / (m - num_S)
        add = np.sum(new)
        if add > 0:
            num_S += np.sum(new)
            compressed[new] = y[new]
            y[new] = 0
            go = True
            
    if norm > 0:
        # Step 2: reorganize entries of y
        B = y[:, None] * A * x[None, :] 
        if proportional:
            B = B / np.sum(np.abs(A), axis = 0)[None, :]
        indices = np.argmax(B, axis = 1)
        Z = np.zeros((d, d))
        Z[range(d), indices] = np.abs(y)/norm*(m - num_S)
    
        # Step 3: systematic resampling    
        mass = np.sum(Z, axis =  0)
        num = np.zeros(d).astype('int')
        u = np.random.uniform()
        j = 0
        upper = mass[0]
        for i in range(m - num_S):
            value = i + u
            while value > upper:
                j += 1
                upper += mass[j]
            num[j] += 1
        
        # Step 4: select entries from each column
        select = np.zeros(d).astype('int')
        for j in range(d):
            if num[j] > 0:
                # Define a lot of stuff
                bottom = np.floor(mass[j])
                r = mass[j] - bottom
                p = Z[:, j]
                if r > 0:
                    q_f = np.copy(p)
                    q_c = np.copy(p)
                    # Fix the probabilities
                    for i in range(d):
                        if np.sum(q_f) > bottom:
                            break
                        if p[i] > 0:
                            if p[i] < r:
                                q_f[i] = 0
                                q_c[i] = p[i]/r
                            else:
                                q_f[i] = (p[i] - r)/(1 - r)
                                q_c[i] = 1
                    q_f[i] += bottom - np.sum(q_f)
                    q_c[i] = (p[i] - q_f[i]*(1 - r))/r
                    if num[j] == bottom:
                        p = q_f
                    else:
                        p = q_c 
                # Systematic resampling
                u = np.random.uniform()
                k = 0
                upper = p[0]
                for i in range(num[j]):
                    value = i + u
                    while value > upper:
                        k += 1
                        upper += p[k]
                    select[k] += 1
        select = select.astype('bool')
        compressed[select] = np.sign(y)[select]*norm/(m - num_S)
    return(compressed)

#%%

# =======
# Old FRI
# =======

def old_compress(y, m):

    x = np.copy(y)
    d = x.size
    compressed = np.zeros(d)

    # Step 1: preserve the largest entries exactly.
    # Usually terminates after a few loops.
    num_S = 0    
    go =  True
    while go:
        go = False
        norm = np.sum(np.abs(x))
        new = np.abs(x) > norm / (m - num_S)
        add = np.sum(new)
        if add > 0:
            num_S += np.sum(new)
            compressed[new] = x[new]
            x[new] = 0
            go = True
            
    if norm > 0:
        # Step 2: systematic resampling    
        mass = np.abs(x) * (m - num_S)/norm
        select = np.zeros(d).astype('int')
        u = np.random.uniform()
        j = 0
        upper = mass[0]
        for i in range(m - num_S):
            value = i + u
            while value > upper:
                j += 1
                upper += mass[j]
            select[j] += 1
        select = select.astype('bool')
        compressed[select] = np.sign(x)[select]*norm/(m - num_S)

    return(compressed)

#%%

#=======================================
# Ex: 2-d diffusion with abs. boundaries
# ======================================

# Define the problem
d = 50
T = 1000
m = 111
A = np.diag(.5*np.ones(d-1), -1) + np.diag(.5*np.ones(d-1), 1)
A = .5*np.kron(np.eye(d), A) + .5*np.kron(A, np.eye(d))
for j in range(np.square(d)):
    if np.sum(A[:, j]) < 1:
        A[:, j] = 0
        A[j,j] = 1
x0 = np.zeros(np.square(d))
x0[np.int(d*(d+1)/2)] = 1.

# Monte Carlo scheme
mc = np.zeros(d**2)
for i in range(m):
    print(i)
    i = np.int(d*(d+1)/2)
    for t in range(T):
        i = np.random.choice(range(d**2), p = A[:,i])
    mc[i] += 1
mc /= m

# Old FRI compression
fri = np.copy(x0)
for t in range(T):
    print(t)
    fri = np.dot(A, fri)
    fri = old_compress(fri, m)
        
# Compression by relative size
relative = np.copy(x0)
for t in range(T):
    print(t)
    relative = compression(A, relative, m)

# Compression by absolute size
absolute = np.copy(x0)
for t in range(T):
    print(t)
    absolute = compression(A, relative, m, proportional = False)

#%%

#=========
# Plotting
# ========

# Compare results
fig, ax = plt.subplots(2, 2, figsize = (12, 10))
pos00 = ax[0, 0].imshow(mc.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[0, 0].tick_params(width = 2)
ax[0, 0].set_title('Independent walkers', font)
pos01 = ax[0, 1].imshow(fri.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[0, 1].tick_params(width = 2)
ax[0, 1].set_title('Old FRI', font)
pos10 = ax[1, 0].imshow(relative.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[1, 0].tick_params(width = 2)
ax[1, 0].set_title('Relative contr. FRI', font)
pos11 = ax[1, 1].imshow(absolute.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[1, 1].tick_params(width = 2)
ax[1, 1].set_title('Absolute contr. FRI', font)
cbar01 = fig.colorbar(pos01)
fig.tight_layout()
fig.savefig('proportional.png', bbox_inches = 'tight', dpi = 200)