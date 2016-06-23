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
import sys
import os

import scipy as sci
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import preprocessing

sys.path.append(os.path.join(os.getcwd(), "fileHandlerPy"))
from prepareSaving import prepareSaving


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
        keys will be sorted using python's sort functionality.

    Returns
    ----------
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
    ----------
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
    fname = prepareSaving(fname, path, ".json")
    pd.Series(mydict).to_json(fname)


def load_dict(fname):
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
    mydict = pd.read_json('test', typ='series').to_dict()
    for key, value in mydict.items():
        mydict[key] = np.array(value)
    return mydict




class milData():
    """
    Class to efficiently handel data for multiple instance learning.


    Attributes
    ----------
    name : string
        The name of the dataset.
    keys : list
        The sorted names of individual bags.

    Methods
    ----------
    get_X_as_dict()
        Return data as dictionary.
    get_X_as_arr()
        Return data as array.
    get_x(n)
        Return one instance of given position.
    get_x_from_B
        Return one instance from a specific bag of given position.
    get_y_as_dict()
        Return instance labels as dictionary.
    get_y_arr()
        Return instance labels as array.
    get_y(n):
        Return one instance label of given position.
    get_y_from_bag(key, l):
        Return one instance label from a specific bag of given position.
    get_z_as_dict(self):
        Return bag labels as dictionary.
    get_z_as_arr(self):
        Return bag labels as array.
    get_z(self, key):
        Return one bag label of given bag.
    add_x(self, x, key):
        Add one instance.
    del_inst(self, n, UPDATE=True):
        Delete one instance.
    del_inst_from_bag(self, key, l, UPDATE=True):
        Delete one instance from a specific bag.
    del_bag(self, key):
        Delete a whole bag.
    safe(self, path):
        Save all MIL data.

    """

    def __init__(self, name, CACHE=True):
        """
        Initialize.

        Parameters
        ----------
        name : string
            The name of the dataset.
        CACHE : bool, optional
            Set if data shall be cached for faster access.
        """
        self.name = name
        self.keys = None  # List of sorted keys of the dicts
        self._X_dict = None  # Dictionary with data per bags
        self._X_arr = None  # numpy.array with data

        self._z_dict = None  # Dictionary with bag labels
        self._z_arr = None  # numpy.array with bag labels as for sorted keys

        self._y_dict = None  # Dictionary with instance labels per bag
        self.__y_arr = None  # numpy.array with instance labels as for sorted key_s

        self._dtype = None  # The numpy.dtype of the instances.
        self._pos_x = None  # Store instance positions for fast access
        self._N_X = 0  # The number of instances stored
        self._N_B = 0  # The number of bags stored
        self._N_D = 0  # The nimension of each feature vector
        self.__CACHE = CACHE

    def get_X_as_dict(self, ):
        """
        Return data as dictionary.

        Returns
        ----------
        dictionary
            The MIL dataset as dictionary. Values are ndarrays.
        """
        return self._X_dict

    def get_X_as_arr(self, ):
        """
        Return data as array.

        Returns
        ----------
        ndarray
            The MIL dataset as one ndarray.
        """
        if self.__CACHE:
            if self._X_arr is None:
                self.__cache()
            return self._X_dict
        else:
            return dict_to_arr(self._X_dict, self.keys)

    def get_B(self, key):
        """
        Return data of one bag.

        Parameters
        ----------
        key : key
            The key of which bag to return.

        Returns
        ----------
        ndarray
            The MIL dataset of one bag.
        """
        return self._X_dict[key]

    def get_x(self, n):
        """
        Return one instance of given position.

        Parameters
        ----------
        n : int
            The overall position of the wanted instance.

        Returns
        ----------
        ndarray
            The dataset of one instance.
        """
        if self._pos_x is None:
            self.__map_x()
        return self._X_dict[self._pos_x[n][0]],self._pos_x[n][1]


    def get_x_from_B(self, key, l):
        """
        Return one instance from a specific bag of given position.

        Parameters
        ----------
        key : key
            The key of the bag to look in.
        l : int
            The local position of the wanted instance.

        Returns
        ----------
        ndarray
            The dataset of one instance.
        """
        return self._X_dict[key][l]

    def get_y_as_dict(self):
        """
        Return instance labels as dictionary.

        Parameters
        ----------
        key : key
            The key of which bag to return.

        Returns
        ----------
        ndarray
            The MIL instance labels as one dictionary.
        """
        return self._y_dict

    def get_y_as_arr(self):
        """
        Return data as array.

        Returns
        ----------
        ndarray
            The MIL dataset as one ndarray.
        """
        if self.__CACHE:
            if self.y_arr is None:
                self.__cache()
            return self._y_dict
        else:
            return dict_to_arr(self._y_dict, self.keys)

    def get_y(self, n):
        """
        Return one instance label of given position.

        Parameters
        ----------
        n : int
            The overall position of the wanted instance label.

        Returns
        ----------
        ndarray
            The label of one instance.
        """
        if self._pos_x is None:
            self.__map_x()
        return self._y_dict[self._pos_x[n][0]],self._pos_x[n][1]

    def get_y_from_bag(self, key, l):
        """
        Return one instance label from a specific bag of given position.

        Parameters
        ----------
        key : key
            The key of the bag to look in.
        l : int
            The local position of the wanted instance.

        Returns
        ----------
        ndarray
            The label of one instance.
        """
        return self._y_dict[key][l]

    def get_z_as_dict(self):
        """
        Return bag labels as dictionary.

        Parameters
        ----------
        key : key
            The key of which bag to return.

        Returns
        ----------
        ndarray
            The MIL bag labels as one dictionary.
        """
        return self._z_dict

    def get_z_as_arr(self):
        """
        Return bag labels as array.

        Returns
        ----------
        ndarray
            The MIL bag labels as one ndarray.
        """
        if self.__CACHE:
            if self._z_arr is None:
                self.__cache()
            return self._z_dict
        else:
            return dict_to_arr(self._z_dict, self.keys)

    def get_z(self, key):
        """
        Return one bag label of given bag.

        Parameters
        ----------
        key : key
            The key of which bag to return.

        Returns
        ----------
        ndarray
            The label of one bag.
        """
        return self._z_dict[key]

    def add_inst(self, key, x, y, z=None,  UPDATE=True):
        """
        Add one instance.

        Parameters
        ----------
        key : key
            The key of which bag to add to.
        x : array_like
            Instance to add. The shape must mach the shape of other instances.
        y : int
            Instance label to add.
        z : int, optional
            Bag label to add. If non is given, the instance label will be used
            for new bags.
        UPDATE : bool, optional
            If True, bag labels will be recalculated to fit MIL constraints.
        """
        self._X_arr = None
        self._y_arr = None
        self._z_arr = None
        self._pos_x = None

        if key in self.keys:
            self._X_dict[key] = np.concatenate((self._X_dict[key], np.array(x)), axis=0)
            self._y_dict[key] = np.concatenate((self._y_dict[key], np.array(y)), axis=0)
            if z is not None:
                self._z_dict[key] = np.array(z)
        else:
            self._X_dict[key] = np.array(x)
            self._y_dict[key] = np.array(y)
            if z is not None:
                self._z_dict[key] = np.array(z)
            else:
                self._z_dict[key] = np.array(y)
            self._N_B += 1
        if UPDATE:
            self._z_dict[key] = np.max(self._y_dict[key])
        self.__sort_keys()
        self._N_X +=1
        self._N_D = len(x)

    def del_inst(self, n, UPDATE=True):
        """
        Delete one instance.

        Parameters
        ----------
        n : int
            The overall postion of the instance to delete.
        UPDATE : bool, optional
            If True, bag labels will be recalculated to fit MIL constraints.
        """

        if self._pos_x is None :
            key, l = find_pos(self._X_dict, n, self.keys)
        else:
            key, l = self._pos_x(n)

        self.del_inst_from_bag(key, l, UPDATE=UPDATE)

    def del_inst_from_bag(self, key, l, UPDATE=True):
        """
        Delete one instance from a specific bag.

        Parameters
        ----------
        key : key
            The key of which bag to add to.
        l : int
            The local position of the wanted instance.
        UPDATE : bool, optional
            If True, bag labels will be recalculated to fit MIL constraints.
        """
        if len(self._X_dict[key]) == 1:
            del self._X_dict[key]
            del self._y_dict[key]
            del self._z_dict[key]
            self._N_B -= 1
        else:
            self._X_dict[key] = np.delete(self._X_dict[key], l, 0)
            self._y_dict[key] = np.delete(self._X_dict[key], l, 0)
            self._z_dict[key] = np.max(self._y_dict[key])
            if UPDATE:
                self._z_dict[key] = np.max(self._y_dict[key])

        self._X_arr = None
        self._y_arr = None
        self._z_arr = None
        self._pos_x = None

        self._N_X -=1
        self.__sort_keys()

    def del_bag(self, key):
        """
        Delete a whole bag.

        Parameters
        ----------
        key : key
            The key of which bag to add to.
        """

        self._N_B -= 1
        self._N_X -= len(self._X_dict[key])

        del self._X_dict[key]
        del self._y_dict[key]
        del self._z_dict[key]

        self._X_arr = None
        self._y_arr = None
        self._z_arr = None
        self._pos_x = None

    def safe(self, path):
        """
        Save all MIL data. Three .json files will be generated: One for the
        data, one for the instance labels and one for the bag labels.

        Parameters
        ----------
        path : string
            The path where to save.
        """
        save_dict(self._X_dict, self.name + "_x", path)
        save_dict(self._y_dict, self.name + "_y", path)
        save_dict(self._z_dict, self.name + "_z", path)

    def load(self, path):

        self._X_dict = load_dict()
        self.__update()
        self.__clear()

    def __sort_keys(self):
        """
        Sort all keys
        """
        self.keys = sorted(self.keys)

    def __cache(self):
        """
        Cache intermediate results for easy acces
        """
        self.X_arr = dict_to_arr(self._X_dict, self.keys)
        self.y_arr = dict_to_arr(self._y_dict, self.keys)
        self.z_arr = dict_to_arr(self._z_dict, self.keys)
        self.__map_x()

        # create dict in which bag to find each instance

    def __map_x(self):
        """
        Map global position of one instance to bag key and bag position.
        """
        self._pos_x = {}
        l = 0  # count along instance in bag
        b = 0  # count along bags
        for i in range(self._N_X):
            if l < len(self._X_dict[self.keys[b]]):
                l += 1
            else:
                b += 1
                l = 0
            self._pos_x[i] = [self.keys[b], l]


