#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:21:24 2018

@author: gabo, zach
"""

import numpy as np
import ast
import json

from collections import Counter
from itertools import product


# The JSON holds tuples
def build_lookup_table(filename):
    with open(filename, 'r') as fp:
        lookup = json.load(fp)

    return lookup


def key_to_array(string):
    return np.array(ast.literal_eval(string))


# Needs to be general
def array_to_key(matrix):
    try:
        return str(tuple(tuple(i) for i in matrix))
    except TypeError:
        return str(matrix)


def calculate_bdm(array, lookup, block=4, dim=2, boundaries='ignore', verbose=False):
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
    assert dim == 2, 'Only matrices are allowed'

    if verbose:
        print('The full matrix (%i,%i) to be decomposed:' % shape)
        print(array)

    # Alter shape for iteration
    shape = np.array(shape) // block
    func = lambda x: range(0,x,1)
    shape = map(func,shape)

    # Leftovers will be ignored
    submatrices = []
    for item in product(*list(shape)):
        sm = array
        for i in range(0,dim):
            sm = np.take(sm,list(range(item[i]*block,(item[i]+1)*block)),axis = i)
        submatrices.append(array_to_key(sm))

    counts = Counter(submatrices)
    bdm_value = sum(lookup[string] + np.log2(n) for string, n in counts.items())
    
    if verbose:
        print('Base submatrices:')
        for s, n in counts.items():
            submatrix = key_to_array(s)
            print(np.array(submatrix), 'n =', n)
        print('BDM calculated value =', bdm_value)
    
    # Just to check whether there were repetitions in the decomposition
    # print('Were all submatrices unique?', set(counts.values()) == {1})

    return bdm_value


if __name__ == '__main__':

    lookup = build_lookup_table('K-4x4.json')
    strings = []
    with open('example_input.txt','r') as input_file:
        for line in input_file.readlines():
            # Removes all kinds of whitespace and can handle edge case where file doesn't terminate with newline char
            strings.append(line.rstrip())

    bdmvals = [calculate_bdm(key_to_array(s), lookup, verbose=True) for s in strings]
    for val in bdmvals:
        # I get the same vals as with the original script
        print(val)
