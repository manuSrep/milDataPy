#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Functions and classes to store and handel MIL datasets efficiently.
:author: Manuel Tuschen
:date: 23.06.2016
:license: FreeBSD

License
----------
Copyright (c) 2016, Manuel Tuschen
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from __future__ import division, absolute_import, unicode_literals, print_function

import scipy as sci
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import preprocessing


def dict_to_arr(mydict, keys=None):
    """
    Convert a MIL dictionary to a 1D ndarray dataset.

    Parameters
    ----------
    mydict : dictionary
        Dictionary with array_like values. Each such array_like must have the
        same shape except for the first dimension.
    keys : key_like, optional
        Only entries of the given keys will be includd in the output array. The
        row ordering will correspond to the keys ordering. If None is given all
        keys will be sortes using python's sort functionality.

    Returns
    -------
    myarr : array_like
        The converted array. Each first axis element corresponds to one
        instance.
    N_b : ndarray
        The number of instances corresponding to each bag.
    keys: list
        The name of the bags after which myarr and N_b was created.
    """
    myarr = None
    N_b = None

    if keys is None:
        keys = sorted(mydict.keys)
        if k == 0:  # we initialize the output array
            myarr = np.array(mydict[key])
            N_b = np.array([len(mydict[key])], dtype=np.int)
        else:
            myarr = np.concatenate((myarr, mydict[key]))
            N_b = np.concatenate(
                (N_b, np.array([len(mydict[key])], dtype=np.int)))
    return myarr, N_b, keys


def arr_to_dic(myarr, keys, N_b):
    """
    Convert an MIL ndarray to an dictionary with each element of the first axis
    becoming one value.

    Parameters
    ----------
    myarr : ndarray
        Ndarray to convert to dictionary.  Each first axis element will
        correspond to one instance.
    keys : list
        keys for each bag to construct the dictionary from. The ordering of the
         keys will define which array entry will belong to which bag.
    N_b : ndarray
        The number of instances corresponding to each bag.

    Returns
    -------
    mydict : dictionary
        Dictionary with array_like values corresponding to the given keys.
    """

    if len(keys) != len(N_b):
        raise ValueError("Incorrect number of keys or bag instance number.")
    if np.sum(N_b) != len(myarr):
        raise ValueError("Incorrect number of instances in array or in bag "
                         "instance number.")

    mydict = dict.fromkeys(keys)
    
    n = 0
    for k,key in enumerate(keys):
        mydict[key] = np.array(myarr[n:n+N_b[k]], dtype=myarr.dtype)
        n += N_b[k]
    return mydict


def find_pos(dict_, k, keys):
    '''
    Find the position of the overall kth element in one set of a dictionary.

    :param dict_: The dictionary to search in.
    :param k: The overall element position to look for.
    :param keys: Odered list of keys to go through.

    :return key, pos: The postion of the kth element.
    '''
    len_ = -1
    for i, key in enumerate(keys):
        len_ += len(dict_[key])

        if len_ == k:
            return (key, len(dict_[key]) - 1)
        elif len_ > k:
            return (key, k - len_ + len(dict_[key]) - 1)
        else:
            continue


def save_dict(mydict, fname, path=None):
    """
    Save the dictionary with MIL data to jason file.

    Parameters
    ----------
    mydict : dictionary
        Dictionary with array_like values. Each such array_like must have the
        same shape except for the first dimension.
    fname: string
        The name of the resulting .json file
    path : string, optional
        The path where to store the datafile.
    """

    f

    pd.Series(mydict).to_json(fname)


def load_dict(fname):
    mydict = pd.read_json('test', typ='series').to_dict()
    for key, value in mydict.items():
        mydict[key] = np.array(value)
    return mydict
