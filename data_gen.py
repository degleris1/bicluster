# Data Generator
# Anthony Degleris
# Last update: August 17th, 2017

# This module contains the functions to generate random matrices with finite values.
# Any discrete distribution is supported by providing a list of possible values and another
# list with corresponding cumulative probabilities.

import numpy as np


# ACCEPTS a list of choices and a list of associated cumulative probabilities.
# RETURNS a randomly selected choice based on the cdf.
# RAISES an exception if no entry is selected (likely choices or cdf is incorrect).
def pick_rand(choices, cdf):
    c = np.random.rand()
    for i in range(0, len(cdf)):
        if (c <= cdf[i]):
            return choices[i]
    raise 'Failure picking an element. Check your cdf.'


# ACCEPTS the matrix height, the matrix width, a list of choices, and a list of
# associated cumulative probabilities.
# RETURNS a matrix with entries based on the choices and cdf.
def create_matrix(n1, n2, choices, cdf):
    M = np.zeros((n1,n2))
    for r in range(0, n1):
        for c in range(0, n2):
            M[r][c] = pick_rand(choices, cdf)
    return M


# ACCEPTS a matrix, the bicluster height, the bicluster width, a list of choices,
# a list of associated cumulative probabilities, the starting row of the cluster,
# and the starting column of the cluster.
# EXECUTION adds a new cluster to the matrix (modifiying input).
# RETURNS the updated matrix.
# RAISES an exception if cluster won't fit.
def add_cluster(M, k1, k2, choices, cdf, s1=0, s2=0):
    if (s1+k1 > len(M) or s2+k2 > len(M[0])):
        raise 'Bicluster too large.'
    for r in range(s1, s1+k1):
        for c in range(s2, s2+k2):
            M[r][c] = pick_rand(choices, cdf)
    return M    
    

    
    
    
    
    