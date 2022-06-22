# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:13:28 2020

@author: Rob
"""

import numpy as np
import matplotlib.pyplot as plt
import os
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

def compression(A, x, m, allocation):

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
        # Step 2a: assign each entry of y to a single index
        nonzero_y = np.arange(d)[y != 0]
        nonzero_x = np.arange(d)[x != 0]
        B = np.sign(y[nonzero_y])[:, None] * A[nonzero_y[:, None], nonzero_x]
        if (allocation=='relative'):
            B *= np.sign(x[nonzero_x])[None,:]
            Bnrms = np.sum(np.abs(B), axis = 0)
            Bnrms[Bnrms == 0] = 1
            B /= Bnrms[None, :]
            # Old version of relative
            # B /= np.sum(np.abs(A[:, nonzero_x]), axis=0)[None,:]
            indices = np.argmax(B, axis = 1)
        elif (allocation=='random'):
            B *= (x[nonzero_x])[None, :]
            nrows = B.shape[0]
            indices = np.zeros(nrows, dtype=int)
            for kk in range(nrows):
                absrow = np.abs(B[kk,:])
                absrow /= np.sum(absrow)
                indices[kk] = np.random.choice(absrow.size, p = absrow)
        else:
            B *= (x[nonzero_x])[None, :]
            indices = np.argmax(B, axis = 1)
        
        
        # Step 2b: assign the y mass to the Z matrix
        mass_y = np.abs(y[nonzero_y]) * (m - num_S)/norm
        Z = np.zeros(B.shape)
        for i in range(nonzero_y.size):
            Z[i, indices[i]] = mass_y[i]
    
        # Step 3: systematic resampling    
        mass = np.sum(Z, axis =  0)
        num = np.zeros(nonzero_x.size).astype('int')
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
        select = np.zeros(nonzero_y.size).astype('int')
        for j in np.arange(nonzero_x.size)[num > 0]:
            # Define a lot of stuff
            bottom = np.floor(mass[j])
            r = mass[j] - bottom
            p = Z[:, j]
            if r > 1e-8:
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
                    if np.sum(q_f) < bottom:
                        break
                q_f[i] += bottom - np.sum(q_f)
                q_c[i] += bottom + 1 - np.sum(q_c)
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
                while value > upper - 1e-8:
                    k += 1
                    upper += p[k]
                select[k] += 1
        selector = nonzero_y[select.astype('bool')]
        compressed[selector] = np.sign(y)[selector]*norm/(m - num_S)
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
        # print(i)
    mc[i] += 1
mc /= m

# Old FRI compression
fri = np.copy(x0)
for t in range(T):
    print(t)
    fri = np.dot(A, fri)
    fri = old_compress(fri, m)
        
# Compression by relative size
relfri = np.copy(x0)
for t in range(T):
    print(t)
    relfri = compression(A, relfri, m, allocation='relative')    

# Compression by absolute size
absfri = np.copy(x0)
for t in range(T):
    print(t)
    absfri = compression(A, absfri, m, allocation = 'absolute')



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
pos10 = ax[1, 0].imshow(relfri.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[1, 0].tick_params(width = 2)
ax[1, 0].set_title('Relative contr. FRI', font)
pos11 = ax[1, 1].imshow(absfri.reshape((d, d)), vmin = 0, vmax = .05, cmap='hot')
ax[1, 1].tick_params(width = 2)
ax[1, 1].set_title('Absolute contr. FRI', font)
cbar01 = fig.colorbar(pos01)
fig.tight_layout()
fig.savefig('proportional.png', bbox_inches = 'tight', dpi = 200)