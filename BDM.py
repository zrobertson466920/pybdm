#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:21:24 2018

@author: gabo, zach
"""

import numpy as np
from collections import Counter

def build_lookup_table(filename):
    with open(filename,'r') as datafile:
        lookup = {}
        for line in datafile.readlines():
            items = line.split(',')
            base_matrix = items[0]
            kvalue = float(items[1])
            lookup[base_matrix] = kvalue
    return lookup


def string_to_nestedlist(string):
    return [[int(s) for s in row] for row in string.split('-')]


def nestedlist_to_string(nestedlist):
    return '-'.join([''.join([str(i) for i in row]) for row in nestedlist])


def calculate_bdm(string, lookup, d=4, boundaries='ignore', verbose=False):
    """
    Input
    --------
    string : string
        binary matrix in string format, where rows are separated by minus signs
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
    assert boundaries == 'ignore', 'Boundary conditions not implemented yet.'
    
    rows = string_to_nestedlist(string)
    if verbose:
        print('The full matrix to be decomposed:')
        print(np.array(rows))
    nrows, ncols = len(rows), len(rows[0])
    
    for row in rows:
        assert len(row) == ncols, '{}'.format()
    
    n_rowblocks = int(nrows / d)
    n_colblocks = int(ncols / d)
    
    # Leftovers will be ignored
        
    submatrices = []
    for i in range(n_rowblocks):
        for j in range(n_colblocks):
            # pick a submatrix in list-of-lists format
            submatrix = [row[d * j : d * (j+1)] for row in rows[d * i : d * (i+1)]]                
            # convert to string format
            string = nestedlist_to_string(submatrix)
            submatrices.append(string)
    counts = Counter(submatrices)
    bdm_value = sum(lookup[string] + np.log2(n) for string, n in counts.items())
    
    if verbose:
        print('Base submatrices:')
        for s, n in counts.items():
            submatrix = string_to_nestedlist(s)
            print(np.array(submatrix), 'n =', n)
        print('BDM calculated value =', bdm_value)
    
    # Just to check whether there were repetitions in the decomposition
    # print('Were all submatrices unique?', set(counts.values()) == {1})

    return bdm_value


if __name__ == '__main__':

    lookup = build_lookup_table('D5.CSV')
    strings = []
    with open('example_input.txt','r') as input_file:
        for line in input_file.readlines():
            # Removes all kinds of whitespace and can handle edge case where file doesn't terminate with newline char
            strings.append(line.rstrip())

    bdmvals = [calculate_bdm(s, lookup, verbose=True) for s in strings]
    for val in bdmvals:
        # I get the same vals as with the original script
        print(val)
