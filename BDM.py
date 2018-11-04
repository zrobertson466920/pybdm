#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:21:24 2018

@author: gabo, zach
"""

import numpy as np
import ast
import json

import matplotlib.pyplot as plt

from collections import Counter
from itertools import product

import scipy.stats


# The JSON holds tuples
def build_lookup_table(filename):
    with open(filename, 'r') as fp:
        lookup = json.load(fp)

    return lookup


def key_to_array(string):
    return np.array(ast.literal_eval(string))


# Needs to be general
def array_to_tuple(matrix):
    try:
        return tuple(array_to_tuple(i) for i in matrix)
    except TypeError:
        return tuple(matrix)


# Partition the data set into chunks
def partition(array, block=4, dim=2, boundaries='ignore', verbose=False):
    """
    Input
    --------
    string : string
        numpy array
    lookup : dict
        key is base matrix, value is its CTM value. Base matrices must be of
        size d x d.
    d : integer
        size of base matrices
        
    Output
    ------
    bdm : float
        BDM value for the matrix
    """

    shape = np.shape(array)
    dim = len(shape)

    assert boundaries == 'ignore', 'Boundary conditions not implemented yet.'
    assert shape.count(shape[0]) == len(shape), 'Only square matrices are allowed.'
    #assert dim == 2, 'Only matrices are allowed'

    if verbose:
        #print('The full matrix (%i) to be decomposed:' % shape)
        print(array)

    # Alter shape for iteration
    tshape = np.array(shape) // block
    func = lambda x: range(0,x,1)
    shape = map(func,tshape)

    # Leftovers will be ignored
    submatrices = np.zeros(tuple(tshape), dtype = "object_")
    for item in product(*list(shape)):
        sm = array
        for i in range(0,dim):
            sm = np.take(sm,list(range(item[i]*block,(item[i]+1)*block)),axis = i)
        submatrices[item] = str(array_to_tuple(sm))

    return submatrices


def calculate_bdm(array, lookup, block=4, dim=2, boundaries='ignore', verbose=False):

    submatrices = partition(array, block, dim, boundaries, verbose)
    counts = Counter(submatrices.flatten())
    bdm_value = sum(lookup[string]+np.log2(n) for string, n in counts.items())
    
    if verbose:
        print('Base submatrices:')
        for s, n in counts.items():
            print(key_to_array(s), 'n =', n)
        print('BDM calculated value =', bdm_value)
    
    # Just to check whether there were repetitions in the decomposition
    # print('Were all submatrices unique?', set(counts.values()) == {1})

    return bdm_value


def calculate_rbdm(array, lookup, block=3, dim=2, boundaries='ignore', verbose=False):

    submatrices = partition(array, block, dim, boundaries, verbose)
    renorm_matrix = np.zeros(np.shape(submatrices))
    for index, value in np.ndenumerate(submatrices):
        renorm_matrix[index] = flatten(key_to_array(value))
    return calculate_bdm(array, lookup, 3, dim, boundaries, verbose)


def flatten(square):
    count = 0
    for val in square:
        if val == 1:
            count += 1

    if count >= np.size(square) / 2:
        return 1
    else:
        return 0


def flip(a):
    idx = np.random.choice(a,replace=False)
    a[idx] = not(a[idx])
    return a


def flips(lookup1):
    flips = []
    for item in lookup1:
        flips.append(lookup1[str(array_to_tuple(flip(key_to_array(item))))])
    return flips


if __name__ == '__main__':

    lookup_base = build_lookup_table('K-3.json')
    lookup_12 = build_lookup_table('K-12m.json')
    strings = list(lookup_12)

    values = [(calculate_rbdm(key_to_array(s), lookup_base, block = 4, dim = 1, verbose=False),lookup_12[s]) for s in strings]
    bdm_vals,ctm_vals = zip(*values)

    pvalues = []
    bdm1 = []
    bdm2 = []
    for s in strings:
        temp = key_to_array(s)
        p_mean = []
        pp_mean = []
        for i in range(0,12):
            p_val = flip(key_to_array(s))
            p_mean.append(calculate_rbdm(p_val, lookup_base, block = 4, dim = 1, verbose=False))
            pp_mean.append(lookup_12[str(array_to_tuple(p_val))])
        pvalues.append((np.mean(p_mean),np.mean(pp_mean)))
    p_bdm, p_ctm = zip(*pvalues)

    bdm_vals = np.array(bdm_vals)
    p_bdm = np.array(p_bdm)
    ctm_vals = np.array(ctm_vals)
    p_ctm = np.array(p_ctm)
    d_bdm = bdm_vals - p_bdm
    d_ctm = ctm_vals - p_ctm

    score = d_bdm * d_ctm
    print(len(score[score > 0]) / len(score))

    plt.figure()
    print(scipy.stats.pearsonr(d_bdm,d_ctm))
    #plt.scatter(ctm_vals,d_ctm, c = 'r')
    plt.scatter(bdm_vals,d_bdm, c = 'b')
    plt.xlabel("BDM Value")
    plt.ylabel("CTM Value")
    plt.title("BDM vs. CTM")

    plt.show()