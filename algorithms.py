# Algorithms
# Anthony Degleris
# Last update: August 17th, 2017

# This module contains the algorithms produced during our research project
# (as well as other algorithms used as benchmarks). 
# Each algorithm should accept a data matrix, half the height of the submatrix 
# (half is used to support split submatrices), and whether or not the submatrix 
# is split. Finally, an algorithm should output the the column indices
# recovered.
# An index of the algorithms is provided to make accessing multiple algorithms 
# easier.
# Note: many of these functions assume proper usage, and will not handle bad
# input.

import numpy as np
import matplotlib.pyplot as plt

ITERATIONS = 20

# ACCEPTS the data to search, the number of indices to return, whether
# to search for the largest or smallest, and if sign should be ignored.
# RETURNS an ordered list of the largest/smallest k indices.
def best_indices(data, k, best='largest', mag_only=False):   
    dtype = [('value', float), ('index', int)]
    if(mag_only):
        ids = [(abs(data[i]), i) for i in range(0, len(data))]
    else:
        ids = [(data[i], i) for i in range(0, len(data))]
    idm = np.array(ids, dtype=dtype)
    idm = np.sort(idm, order='value')
  
    if(best == 'largest'):
        return sorted(list(idm['index'][len(data)-k:]))
    else:
        return sorted(list(idm['index'][:k]))

    
# ACCEPTS a matrix, half the height of the submatrix, a row
# estimate, and a column estimate.
# EXECUTION iteratively uses the previous row/column estimate
# to sum part of the matrix and create a new row/column estimate.
# RETURNS the columns recovered.
def recovery_full(M, k, rows=None, cols=None):
    if(rows == None):
        rows = best_indices(np.sum(M[:,cols], axis=1), 2*k)
    for i in range(0, ITERATIONS):
        cols = best_indices(np.sum(M[rows,:], axis=0), 2*k)
        rows = best_indices(np.sum(M[:,cols], axis=1), 2*k)
    return cols


# ACCEPTS a matrix, half the height, each cluster's row estimate,
# and a column estimate.
# EXECUTION iteratively uses the previous row/column estimate
# to sum part of the matrix and create a new row/column estimate.
# RETURNS the columns recovered.
def recovery_split(M, k, rows1=None, rows2=None, cols=None):
    if(rows1 == None or rows2 == None):
        rsums = np.sum(M[:,cols], axis=1)
        rows1 = best_indices(rsums, k, 'largest')
        rows2 = best_indices(rsums, k, 'smallest')
    for i in range(0, ITERATIONS):
        csums = np.sum(M[rows1,:], axis=0) - np.sum(M[rows2,:], axis=0)
        cols = best_indices(csums, 2*k)
        rsums = np.sum(M[:,cols], axis=1)
        rows1 = best_indices(rsums, k, 'largest')
        rows2 = best_indices(rsums, k, 'smallest')
    return cols
        

# ACCEPTS a matrix, half the height of the submatrix, and whether to
# display graphics.
# EXECUTION sums the rows of the matrix, choosing the best indices
# by their rows.
# RETURNS the column indices recovered.
def counting(M, k, split=False, graphics=False):
    if(split):
        rsums = np.sum(M, axis=1)
        rows1 = best_indices(rsums, k, 'largest')
        rows2 = best_indices(rsums, k, 'smallest')
        return recovery_split(M, k, rows1, rows2, None)
    else:
        return recovery_full(M, k, best_indices(np.sum(M, axis=1), 2*k), None)
    
    
    
    
run_alg = {'counting':counting}