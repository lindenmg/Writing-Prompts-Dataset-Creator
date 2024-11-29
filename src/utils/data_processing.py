# Copyright 2024 Gabriel Lindenmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

import collections
import hashlib
import itertools
import json

import torch


class DataProcessor:

    #  _____                     __
    # |_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___
    #   | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \
    #   | || | | (_| | | | \__ \  _| (_) | |  | | | | | |
    #   |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|

    @staticmethod
    def change_type_of_dict_key(dict_, new_type=str):
        """
        Parameters
        ----------
        dict_: dict
        new_type: type

        Returns
        -------
        dict
            Dictionary whose keys are now of type ``new_type``
            - As long as the original type is convertible into
            the new one
        """
        if not isinstance(dict_, dict):
            raise TypeError("``dict_`` must be of type ``dict``"
                            ", but is {:}".format(type(dict_)))
        if not isinstance(type(new_type), type):
            raise TypeError("``new_type`` has to be of type ``type``"
                            ", but is {:}".format(type(new_type)))
        return {new_type(key): value for key, value in dict_.items()}

    @staticmethod
    def conv_inner_to_tensor(array, tensor_type=torch.tensor):
        """
        Converts a list-like of list-like to a list of Tensors

        Parameters
        ----------
        array: list, np.ndarray
            Containing list-like of compatible data types for Tensor
        tensor_type: torch.Tensor
            Which type of tensor it should be
        Returns
        -------
        list of torch.Tensor
            A list of Tensor which values and shape matches the array
            The rows (axis 0) are now Tensors, stored in a list object
        """
        return [tensor_type(a) for a in array]

    @staticmethod
    def flatten(list_of_lists):
        """
        Flattens a list which contains lists (list/scalar-mix) to a list

        Parameters
        ----------
        list_of_lists : list of list

        Returns
        -------
        list
            contains the Variables that have been in the nested inner lists
        """
        return list(itertools.chain.from_iterable(list_of_lists))

    @staticmethod
    def filter_dict(d_dict_, keys):
        """
        Filters out all key-value pairs from a dict that are not in the ``keys`` list.

        Parameters
        ----------
        d_dict_: dict
            the dict that shall be filtered
        keys: list of strings
            the keys, that shall remain in the dict

        Returns
        -------
        dict
            the dictionary with only the keys provided in the 'keys' parameter
        """
        return {k: v for k, v in filter(lambda t: t[0] in keys, d_dict_.items())}

    #  ____
    # |  _ \ _ __ ___   ___ ___  ___ ___
    # | |_) | '__/ _ \ / __/ _ \/ __/ __|
    # |  __/| | | (_) | (_|  __/\__ \__ \
    # |_|   |_|  \___/ \___\___||___/___/

    @staticmethod
    def idx_lookup_from_list(list_, default_dict=False, default_idx=0):
        """
        Creates a dict with the list elements as key and their index as value

        Parameters
        ----------
        list_: list
        default_dict:bool
            If it shall construct a dict which returns 0
            if the key does not exist in the dictionary
        default_idx: int
            Default index of the ``default_dict``

        Returns
        -------
        dict or collections.defaultdict

        Raises
        ------
        Warning, when the values in list_ are not unique
        """
        if default_dict:
            look_up = collections.defaultdict(lambda: default_idx)
        else:
            look_up = {}

        len_list = len(list_)
        len_unique = len(list(set(list_)))
        if len_list != len_unique:
            raise Warning("The values in list_ are not unique!")

        for i, w in enumerate(list_):
            look_up[w] = i
        return look_up

    @staticmethod
    def hash_dict(dictionary):
        """
        Creates a sha1 hash of a dict

        Parameters
        ----------
        dictionary: dict
            the dict for which a hash shall be created
        Returns
        -------
        str
            the sha1 hash in hexadecimal utf-8 encoding of the parameter ``dictionary``
        """
        j_args = json.dumps(dictionary, ensure_ascii=True, sort_keys=True)
        sha = hashlib.sha256()
        sha.update(j_args.encode(encoding='utf-8'))
        return sha.hexdigest()
