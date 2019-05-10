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
import pylab

from collections import Counter
from itertools import product

import scipy.stats as stats


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
    block : integer
            size of base matrices
    dim : integer
        1 is a list
        2 is a matrix
        etc...
        
    Output
    ------
    submatrices : list
        partition form of the input
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
    """
        Input
        --------
        array : numpy array
            numpy array of 0s and 1s
        lookup : dict
            key is base matrix, value is its CTM value. Base matrices must be of
            size d x d.
        block : integer
            size of base matrices
        dim : integer
            1 is a list
            2 is a matrix
            etc...

        Output
        ------
        bdm : float
            BDM value for the input
        """

    submatrices = partition(array, block, dim, boundaries, verbose)
    counts = Counter(submatrices.flatten())
    #bdm_value = sum(lookup[string]*(1+np.log2(n)) for string, n in counts.items())
    a = len(array[array == 0])
    b = len(array[array == 1])
    ent = -(a * np.log(a/(a+b)+0.01)+b*np.log(b/(a+b)+0.01))
    w = 1
    bdm_value = ent*(1-w)+w*sum(lookup[string] - entropy(string) for string, n in counts.items())
    #bdm_value = (bdm_value - 26)/2

    if verbose:
        print('Base submatrices:')
        for s, n in counts.items():
            print(key_to_array(s), 'n =', n)
        print('BDM calculated value =', bdm_value)

    # Just to check whether there were repetitions in the decomposition
    # print('Were all submatrices unique?', set(counts.values()) == {1})

    return bdm_value


def entropy(string):
    array = key_to_array(string)
    a = len(array[array == 0])
    b = len(array[array == 1])
    return(-(a * np.log(a/(a+b)+0.01)+b*np.log(b/(a+b)+0.01)))


def calculate_rbdm(array, lookup, block=3, dim=2, boundaries='ignore', verbose=False):

    submatrices = partition(array, block, dim, boundaries, verbose)
    renorm_matrix = np.zeros(np.shape(submatrices), dtype = int)
    for index, value in np.ndenumerate(submatrices):
        renorm_matrix[index] = flatten(key_to_array(value))
    return calculate_bdm(renorm_matrix, lookup, len(array) // block, dim, boundaries, verbose)


def bit(string):
    temp = key_to_array(string)
    return(len(temp[temp == 0]))


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


def variance(bdm1):
    x = bdm1/12
    a = -(2*x**4-4*x**3-x**2+3*x-1)
    b = x**2*(x-1)**2
    return 3*np.sqrt(b/a)


if __name__ == '__main__':

    # Build Lookup Tables
    lookup_base = build_lookup_table('data/K-6.json')
    lookup_12 = build_lookup_table('data/K-12m.json')
    # Test String
    strings = ["000000000000"]

    # calculate_bdm expects an array for input
    values = [(s,calculate_bdm(np.array(list(s),dtype = int), lookup_base, block = 6, dim = 1, verbose=False)) for s in strings]
    print(values)