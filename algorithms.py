'''
Algorithms
Anthony Degleris
Last update: August 18th, 2017

This module contains the algorithms produced during our research project
(as well as other algorithms used as benchmarks). 
Each algorithm should accept a data matrix, half the height of the submatrix 
(half is used to support split submatrices), and whether or not the submatrix 
is split. Finally, an algorithm should output the the column indices
recovered.
An index of the algorithms is provided to make accessing multiple algorithms 
easier.
Note: many of these functions assume proper usage, and will not handle bad
input.
'''

import numpy as np
import scipy.sparse.linalg as linear
import matplotlib.pyplot as plt

ITERATIONS = 20


'''
ACCEPTS the data to search, the number of indices to return, whether
to search for the largest or smallest, and if sign should be ignored.
RETURNS an ordered list of the largest/smallest k indices.
'''
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


'''
ACCEPTS a matrix, half the height of the bicluster, the row estimates,
and whether the bicluster is split.
RETURNS an estimate of the columns.
'''
def rows_to_cols(M, k, rows1, rows2=None, split=False):
    if (split):
        csums = np.sum(M[rows1,:], axis=0) - np.sum(M[rows2,:], axis=0)
        return best_indices(csums, 2*k)
    else:
        return best_indices(np.sum(M[rows1,:], axis=0), 2*k)
    

'''
ACCEPTS a matrix, half the height of the bicluster, the column estimate,
and whether the bicluster is split.
RETURNS an estimate of the rows.
'''
def cols_to_rows(M, k, cols, split=False):
    if (split):
        rsums = np.sum(M[:,cols], axis=1)
        return (best_indices(rsums, k, 'largest'), best_indices(rsums, k, 'smallest'))
    else:
        return best_indices(np.sum(M[:,cols], axis=1), 2*k)



'''
ACCEPTS a matrix, half the height of the submatrix, a row
estimate, and a column estimate.
EXECUTION iteratively uses the previous row/column estimate
to sum part of the matrix and create a new row/column estimate.
RETURNS the columns recovered.
'''
def recovery_full(M, k, rows=None, cols=None):
    if(rows == None):
        rows = cols_to_rows(M, k, cols)
    for i in range(0, ITERATIONS):
        cols = rows_to_cols(M, k, rows)
        rows = cols_to_rows(M, k, cols)
    return cols


'''
ACCEPTS a matrix, half the height, each cluster's row estimate,
and a column estimate.
EXECUTION iteratively uses the previous row/column estimate
to sum part of the matrix and create a new row/column estimate.
RETURNS the columns recovered.
'''
def recovery_split(M, k, rows1=None, rows2=None, cols=None):
    if(rows1 == None or rows2 == None):
        rows1, rows2 = cols_to_rows(M, k, cols, True)
    for i in range(0, ITERATIONS):
        cols = rows_to_cols(M, k, rows1, rows2, True)
        rows1, rows2 = cols_to_rows(M, k, cols, True)
    return cols
        

'''
ACCEPTS a matrix, half the height of the bicluster, whether the
bicluster is split, and whether to display graphics.
EXECUTION sums the rows of the matrix, choosing the best indices
by their rows.
RETURNS the column indices recovered.
'''
def counting(M, k, split=False, graphics=False):
    if(split):
        rsums = np.sum(M, axis=1)
        rows1 = best_indices(rsums, k, 'largest')
        rows2 = best_indices(rsums, k, 'smallest')
        return recovery_split(M, k, rows1, rows2, None)
    else:
        return recovery_full(M, k, best_indices(np.sum(M, axis=1), 2*k), None)
    

'''
ACCEPTS a matrix, half the height of the bicluster, whether the 
bicluster is split, and whether to display graphics.
EXECUTION uses singular value decomposition to estimate the columns
of the bicluster.
RETURNS the column indices recovered.
'''
def spectral(M, k, split=False, graphics=False):
    u, s, vt = linear.svds(M, 1)
    if (split):
        return recovery_split(M, k, None, None, best_indices(vt[0,:], 2*k, mag_only=True))
    else:
        return recovery_full(M, k, None, best_indices(vt[0,:], 2*k, mag_only=True))
    

'''
ACCEPTS a matrix, half the height of the bicluster, and whether to
display graphics.
See parent function for execution details.
'''
def recursive_spectral_helper(M, k, graphics):
    u, s, vt = linear.svds(M, 1)
    n = len(M)
    if (n <= 5*k):
        return best_indices(vt[0,:], 2*k, mag_only=True)
    else:
        cols = best_indices(vt[0,:], int(9*n/10), mag_only=True)
        M = M[:, cols]
        u, s, vt = linear.svds(M, 1)
        rows = best_indices(u[:,0], int(9*n/10), mag_only=True)
        return [cols[i] for i in recursive_spectral_helper(M[rows,:], k, graphics)]
    
    
'''
ACCEPTS a matrix, half the height of the bicluster, whether the bicluster
is split, and whether to display graphics.
EXECUTION recursively removes columns and rows using SVD to estimate
the bicluster.
RETURNS an estimate of the columns.
'''
def recursive_spectral(M, k, split=False, graphics=False):
    M_copy = M.copy()
    cols = recursive_spectral_helper(M_copy, k, graphics)
    if (split):
        return recovery_split(M, k, None, None, cols)
    else:
        return recovery_full(M, k, None, cols)
    

'''
ACCEPTS a matrix, half the height of the bicluster, whether the
bicluster is split, and whether to display graphics.
See parent function for execution details.
'''
def recursive_counting_helper(M, k, split, graphics):
    n = len(M)
    rsums = np.sum(M, axis=1)
    if (n <= 5*k):
        return counting(M, k, split, graphics)
    if (split):
        rows1 = best_indices(rsums, int(n/2)-1, 'largest')
        rows2 = best_indices(rsums, int(n/2)-1, 'smallest')
        rows = sorted(rows1 + rows2)
        cols = rows_to_cols(M, len(rows), rows1, rows2, split)
    else:
        rows = best_indices(rsums, n-2, 'largest')
        cols = rows_to_cols(M, n-2, rows, None, split)
    return [cols[i] for i in recursive_counting_helper(M[rows,:][:,cols], k, split, graphics)]
    

'''
ACCEPTS a matrix, half the height of the bicluster, whether the
bicluster is split, and whether to display graphics.
EXECUTION recursively removes rows and columns through enumeration
to estimate the bicluster.
RETURNS the column indices recovered.
'''
def recursive_counting(M, k, split=False, graphics=False):
    M_copy = M.copy()
    cols = recursive_counting_helper(M_copy, k, split, graphics)
    if (split):
        return recovery_split(M, k, None, None, cols)
    else:
        return recovery_full(M, k, None, cols)
    

run_alg = {'cnt':counting, 'spc':spectral, 'rec_spc':recursive_spectral, 'rec_cnt':recursive_counting}