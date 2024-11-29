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

from pathlib import Path
from subprocess import call

import pandas as pd

from src.utils.data_processing import DataProcessor as dp
from src.utils.settings import Config


class HdfsCacher:

    # ToDo: Complete doc comment
    def __init__(self, d_params=None):
        """
        HDFS caching of pd.Dataframes. Two modes: decorator or normal function invocations

        Parameters
        ----------
        d_params: dict or None
            Leave ``None`` if you don't want to use the decorator.
            Else ...
        """
        # HDFS caching part:
        path_prepped = Path(Config.path.data_folder) / "processed" / "prepped.h5"
        path_inter = Path(Config.path.data_folder) / "interim" / "inter.h5"
        self.d_params = d_params

        # Indicators for the preprocessing stages in the HDFS file names
        self.l_legal_stages = ['i', 'p']

        # We use them to write to / load from the respective HDFS
        self.hdfs_inter = pd.HDFStore(str(path_inter), mode='a')
        self.hdfs_prepp = pd.HDFStore(str(path_prepped), mode='a')

        # We use this command-line command to compress the HDFS afterward
        self.l_hdfs_finish = [
            "ptrepack"
            , "--overwrite-nodes"
            , "--chunkshape=auto"
            , "--complevel=3"
            , "--complib=blosc:zstd"
        ]

    # ToDo: Remove necessity of unique pipeline stage names
    # ToDo: Remove necessity of passing key-name in pipeline config file
    def cache(self, func, check_cache=True, operation='', *args, **kwargs):
        """
        Decorator for automated caching and retrieval of processed data

        Parameters
        ----------
        func: function
            Function pointer to function which will be wrapped
        check_cache: bool
            True, if the cache shall be checked for already stored data
            with matching arguments, False else
        operation: str
            Key in config file which determines parameters of this pipeline stage
        args: tuple
        kwargs: dict

        Returns
        -------
        object
            From function behind ``func`` processed data.
        """
        d_task_params = self.d_params.get(operation, None)
        if d_task_params is None:
            return ValueError(f"Key {operation} does not exist in parameter config file!")
        d_task_params, _ = d_task_params

        function_name = d_task_params.get('function', '')
        if function_name != func.__name__:
            return ValueError("Function names from config file and passed "
                              "function pointer don't match! "
                              f"{function_name} != {func.__name__}")

        stage = d_task_params.get('stage', 'p')

        if check_cache:
            result = self.load_hfs_cache(operation, d_task_params, stage)
        else:
            result = None

        if result is None:
            result = func(*args, **kwargs)
            self.append_hdfs_cache(result, operation, d_task_params, stage)

        return result

    def __check_hdfs_params(self, df_data, task: str, d_sub_settings, stage: str):
        if stage not in self.l_legal_stages:
            raise ValueError(f"``stage`` should have value in {self.l_legal_stages}"
                             f", but equals {stage}")
        if not isinstance(d_sub_settings, dict):
            raise TypeError("``d_sub_settings`` must be of type dict"
                            ", but is {:}".format(type(d_sub_settings)))
        if not isinstance(task, str):
            raise TypeError("``task`` must be of type str"
                            ", but is {:}".format(type(task)))
        if df_data is not None and not hasattr(df_data, "to_hdf"):
            raise TypeError("``df_data`` has to be a type with HDFS saving "
                            "capability and compatible with the Pandas data "
                            "objects! But it is of type {:}".format(type(df_data)))

    def __prepare_hdfs_operation(self, task: str, d_sub_settings, stage: str):
        hash_ = dp.hash_dict(d_sub_settings)
        file = f"{task}__{stage}__{hash_}"
        len_file = len(file)

        if len_file > 255:  # ToDo: FIX outdated error message
            raise ValueError(f"Length of HDFS filename ``file`` == {file} "
                             f"is with {len_file} greater than 255, which "
                             f"is not allowed. Please check your JSON "
                             f"configuration file for the Parameter class.")

        store = self.hdfs_inter if stage == 'i' else self.hdfs_prepp
        return file, store

    def load_hfs_cache(self, task: str, d_sub_settings, stage: str):
        """
        Loads data if given HDFS has implied data

        The key ('file-name') is based on the task,
        stage and the hash of the settings dict

        Parameters
        ----------
        task: str
            The current sub-task as short, descriptive name
        d_sub_settings: dict
            Currently relevant settings from the Parameters class or elsewhere
        stage: str
            Either 'i' for interim or 'p' for processed

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame, IF found, ELSE None
        """
        cached = self.check_hdfs_cache(task, d_sub_settings, stage)

        if cached:
            file, store = self.__prepare_hdfs_operation(task, d_sub_settings, stage)
            df_data = store[file]
        else:
            df_data = None
        return df_data

    def check_hdfs_cache(self, task: str, d_sub_settings, stage: str):
        """
        Checks if given HDFS has implied data

        The key ('file-name') is based on the task,
        stage and the hash of the settings dict

        Parameters
        ----------
        task: str
            The current sub-task as short, descriptive name
        d_sub_settings: dict
            Currently relevant settings from the Parameters class or elsewhere
        stage: str
            Either 'i' for interim or 'p' for processed

        Returns
        -------
        bool
            True, IF DataFrame has been found, ELSE False
        """
        self.__check_hdfs_params(None, task, d_sub_settings, stage)
        cached = False
        file, store = self.__prepare_hdfs_operation(task, d_sub_settings, stage)

        if file in store.root:
            cached = True
        return cached

    def append_hdfs_cache(self, df_data, task: str, d_sub_settings
                          , stage: str, rows_total: int):
        """
        Appends HDFS file with ``df_data``. Creates it, if non-existent yet

        The key ('file-name') is based on the task,
        stage and the hash of the settings dict

        Parameters
        ----------
        df_data: pd.DataFrame
        task: str
            The current sub-task as short, descriptive name
        d_sub_settings: dict
            Currently relevant settings from the Parameters class or elsewhere
        stage: str
            Either 'i' for interim or 'p' for processed
        rows_total: int
            The expected number of rows the whole table will have at the end
            after all append operations have been completed
        """
        self.__check_hdfs_params(df_data, task, d_sub_settings, stage)
        file, store = self.__prepare_hdfs_operation(task, d_sub_settings, stage)
        store.append(key=file, value=df_data, index=True, expectedrows=rows_total)
        # ToDo: ==> store.append(key=file, value=df_data, index=True, format='table'
        #          , expectedrows=rows_total, table=True
        #              , data_columns=['prompt_idx', 'story_idx', 'sentence_idx'])
        store.flush(fsync=True)

    def finalize_hdfs_caching(self, task: str, d_sub_settings: dict, stage: str):
        """
        Finishes HDFS file storage. Compress file-store

        The key ('file-name') is based on the task,
        stage and the hash of the settings dict

        Parameters
        ----------
        task: str
            The current sub-task as short, descriptive name
        d_sub_settings: dict
            Currently relevant settings from the Parameters class or elsewhere
        stage: str
            Either 'i' for interim or 'p' for processed
        """
        self.__check_hdfs_params(None, task, d_sub_settings, stage)
        _, store = self.__prepare_hdfs_operation(task, d_sub_settings, stage)

        store.flush(fsync=True)
        store.close()
        origin_f = Path(store.filename)
        backup_f = origin_f.parent / ('backup_' + origin_f.name)
        backup_f = str(backup_f)
        origin_f = str(origin_f)
        call(['mv', origin_f, backup_f])  # ToDo: ADD option to compress without backup to save memory
        l_ptrepack = self.l_hdfs_finish + [backup_f, origin_f]
        call(l_ptrepack)
        call(['rm', backup_f])
        store.open()

    def clean_hdfs(self):
        """
        Closes the opened HDFS connections. Run it after all operations
        """
        if self.hdfs_inter.is_open:
            self.hdfs_inter.close()
        if self.hdfs_prepp.is_open:
            self.hdfs_prepp.close()
