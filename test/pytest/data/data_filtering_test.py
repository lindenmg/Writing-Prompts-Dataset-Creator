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

import os
from pathlib import Path

import pandas as pd

path_test_data = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


def load_data(file: str):
    path_file = Path(path_test_data) / file

    if not path_file.is_file():
        raise FileNotFoundError(f"{str(path_file)}")

    return pd.read_csv(path_file, keep_default_na=False)


class TestDataFilterer:

    @staticmethod
    def check_series(test, gold):
        for t, g in zip(test, gold):
            # print()
            # print("DONE    :" + str(t) + "|||")
            # print("FILTERED:" + str(g) + "|||")
            # print()
            assert t == g

    def test_calc_stats_i(self):
        return NotImplementedError

    def test_drop_stories_i(self):
        return NotImplementedError

    def test_calc_stats_ii(self):
        return NotImplementedError

    def test_drop_stories_ii(self):
        return NotImplementedError

    def test_drop_prompts(self):
        return NotImplementedError

    def test_filter_data(self):
        return NotImplementedError
