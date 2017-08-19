# Testing Interface
# Anthony Degleris
# Last update: August 17th, 2017

# This module contains the the necessary functions to generate large amounts of
# matrix data and test multiple algorithms at once.

import numpy as np
import time
import data_gen as gen



# ACCEPTS the bicluster height, the number of trials for each matrix size, the number
# of matrix sizes, the starting matrix size, the ending matrix size, and the error
# rate in the cluster.
# EXECUTION generates [trials] matrices at every resolution, spaced evenly
# between nstart and nend. Each matrix is generated twice: once with a full cluster
# and once with a split cluster. The distributions used are:
#     noise: choices (-1, 1), cdf (0.5, 1)
#     cluster1: choices (-1, 1), cdf (e, 1)
#     cluster2: choices (-1, 1), cdf (1-e, 1)
# RETURNS a list of the matrix sizes and a dictionary containing the matrices.
def load_matrix_batch(k=10, trials=10, resolution=20, nstart=0.5, nend=1, e=0):
    choices = (-1, 1)
    cdf = {'noise':(0.5, 1), 'cluster1':(e, 1), 'cluster2':(1-e, 1)}
    
        
    nstart = int((nstart)*((2*k)**2))
    nend = int((nend)*(2*k)**2)   
    nlist = [int(i) for i in np.linspace(nstart, nend, resolution)]
    
    data = {}
    time_start = time.clock()
    print('Generating data')
    for n in nlist:
        print('.', end='')
        data[n] = {'full':[], 'split':[]}
        for i in range(0, trials):
            M = gen.create_matrix(n, n, choices, cdf['noise'])
            data[n]['full'].append(gen.add_cluster(M, 2*k, 2*k, choices, cdf['cluster1']))
            data[n]['split'].append(gen.add_cluster(M.copy(), k, 2*k, choices, cdf['cluster2'], k))
    print(' Done in ' + str((time.clock() - time_start)/60) + ' minutes.')
    return nlist, data
            
