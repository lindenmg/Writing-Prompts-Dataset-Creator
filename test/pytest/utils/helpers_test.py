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

import numpy as np

from src.utils.helpers import round_to_next_multiple


class TestHelpers:

    def test_round_to_next_multiple(self):
        upper_fact = 2049
        upper_num = int(4e9)
        size_ = 1000_000
        number = np.random.randint(low=0, high=upper_num, size=size_)
        factor = np.random.randint(low=1, high=upper_fact, size=size_)

        for number, f in zip(number, factor):
            multiple = round_to_next_multiple(number, f)
            assert multiple >= number
            assert f > (multiple - number)
            assert multiple % f == 0
            assert multiple == round(multiple)
