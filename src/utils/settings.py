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

import json
import os

from src.utils.helpers import dict_raise_on_duplicates


class _SubConf:
    pass


class Settings(type):
    _instances = {}

    def __init__(cls, name, bases, attrs, path=None):
        super().__init__(name, bases, attrs)

        cls.config_dict = {}
        cls.load_json(path=path)

    def load_json(self, path):
        """
        For manually loading a config file, which must be in json format.

        Every key in the json is becoming a member of Config with the corresponding value.
        It raises a warning, if the json has duplicate keys.
        Parameters
        ----------
        path: str
            The path to the config json
        """
        self.__remove_config_members()

        with open(path, mode='r') as f:
            self.config_dict = json.load(f, object_pairs_hook=dict_raise_on_duplicates)

        self.__add_config_members(self.config_dict, self)

    def __remove_config_members(self):
        for k, _ in self.config_dict.items():
            if k in self.__dict__.keys():
                delattr(self, k)

    def __add_config_members(self, dictionary, obj):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = self.__add_config_members(v, _SubConf())
            setattr(obj, k, v)
        return obj


class _ConfigSingleton(Settings):
    """
    Parent class for Config (See child at the bottom of the file)
    """

    _instances = {}

    def __init__(cls, name, bases, attrs):
        path = os.path.realpath(os.path.join(__file__
                                             , '..'
                                             , '..'
                                             , '..'
                                             , 'config'
                                             , 'config.json'))

        super().__init__(name, bases, attrs, path)

    def __call__(cls, name, bases, attrs, path=None):
        if cls not in cls._instances:
            cls._instances[cls] = super(_ConfigSingleton, cls).__call__(name
                                                                        , bases
                                                                        , attrs
                                                                        , path=path)
        return cls._instances[cls]


class Config(metaclass=_ConfigSingleton):
    """
    Singleton class for configuration files.

    A configuration file should be provided in the working directory,
    which should be the project root folder.

    Examples
    ---------
        >>> from src.utils.settings import Config

        >>> # configuration parameter access directly by member
        >>> Config.n_cpu

        >>> # alternatively
        >>> Config.config_dict['n_cpu']

        >>> # use alternative config file
        >>> Config.load_json(path='test/test_data/config_test.json'
        >>> param_1 = Config.param_1
    """
    pass
