'''
Testing Interface
Anthony Degleris
Last update: August 19th, 2017

This module contains the the necessary functions to generate large amounts of
matrix data and test multiple algorithms at once.
'''

import numpy as np
import time
import data_gen as gen
import algorithms as alg


'''
ACCEPTS half the bicluster height, the number of trials for each matrix size, the number
of matrix sizes, the starting matrix size, the ending matrix size, and the error
rate in the cluster.
EXECUTION generates [trials] matrices at every noise resolution, spaced evenly
between nstart and nend. Each matrix is generated twice: once with a full cluster
and once with a split cluster. The distributions used are:
    noise: choices (-1, 1), cdf (0.5, 1)
    cluster1: choices (-1, 1), cdf (e, 1)
    cluster2: choices (-1, 1), cdf (1-e, 1)
RETURNS a list of the matrix sizes and a dictionary containing the matrices.
'''
def load_matrix_nbatch(k=10, trials=10, resolution=20, nstart=0.5, nend=1, e=0):
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
            

'''
ACCEPTS half the bicluster height, the matrix size, the number of points to sample,
the number of trials, the starting error, and the ending error.
EXECUTION generates [trials] matrices at every error resolution, spaced evenly
between estart and eend. Each matrix is generated twice: once with a full cluster
and once with a split cluster. The distributions used are:
    noise: choices (-1, 1), cdf (0.5, 1)
    cluster1: choices (-1, 1), cdf (e, 1)
    cluster2: choices (-1, 1), cdf (1-e, 1)
RETURNS a list of the errors and a dictionary containing the matrices.
'''
def load_matrix_ebatch(k=10, n=200, resolution=20, trials=10, estart=0, eend=0.1):
    choices = (-1, 1)
    cdf_noise = (0.5,1)
    
    elist = [i for i in np.linspace(estart, eend, resolution)]
             
    data = {}
    time_start = time.clock()
    print('Generating data')
    for e in elist:
        print('.', end='')
        data[e] = {'full':[], 'split':[]}
        for i in range(0, trials):
            M = gen.create_matrix(n, n, choices, cdf_noise)
            data[e]['full'].append(gen.add_cluster(M, 2*k, 2*k, choices, (e,1)))
            data[e]['split'].append(gen.add_cluster(M.copy(), k, 2*k, choices, (1-e,1)))
    print(' Done in ' + str((time.clock() - time_start)/60) + ' minutes.')
    return elist, data


'''
ACCEPTS a function code (see algorithms module), half the size
of the bicluster, the testing points (noise or error), the
testing data, and whether or not the data is split.
EXECUTION runs the algorithm on all the matrices in the data,
recording its performance and runtime.
RETURNS the performances and their runtimes.
'''
def test_algorithm(func_code, k, pts, data, split=False):
    tests = {}
    times = {}
    
    print('Testing ' + func_code, end='')
    time_start = time.clock()
    for pt in pts:
        print('.', end='')
        tests[pt] = {'split':[], 'full':[]}
        times[pt] = {'split':[], 'full':[]}
        
        for bic_type in (('split', True), ('full', False)):
            for matrix in data[pt][bic_type[0]]:
                truth = set(range(0, 2*k))
                start = time.clock()            
                estimate = alg.run_alg[func_code](matrix, k, bic_type[1])
                times[pt][bic_type[0]].append(time.clock() - start)
                tests[pt][bic_type[0]].append(len(truth.intersection(estimate)) / (2*k))
    print(' Done in ' + str((time.clock() - time_start)/60) + ' minutes.')
    return tests, times