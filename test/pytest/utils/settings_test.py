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

import os
import warnings

from src.utils.settings import Config

path_test_data = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


class TestConfig:

    def test_init(self):
        """
        The test ensures that the default config (<project_root>/config/config.json)
        is loaded and can be accessed with 'Config.config'
        """
        assert 'config_dict' in Config.__dict__.keys()

    def test_load_json(self):
        """
        For loading a different config file
        """
        path_config = os.path.join(path_test_data, 'config_test.json')
        Config.load_json(path_config)
        assert Config.config_dict.get('p_1') == 42

    def test_automatic_member_injection(self):
        path_config = os.path.join(path_test_data, 'config_test.json')
        Config.load_json(path_config)
        assert Config.debug_mode
        assert Config.p_1 == 42
        assert Config.p_2 == 27
        assert Config.subconf_1.p_4 == 25
        assert Config.subconf_1.subconf_2.p_5 == 28
        assert 3 in Config.subconf_1.subconf_2.list_1

        raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                path_config = os.path.join(path_test_data, 'config_test_bad.json')
                Config.load_json(path_config)
            except Warning:
                raised = True
        assert raised
