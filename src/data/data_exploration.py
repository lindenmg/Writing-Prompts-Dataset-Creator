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
import sys

# In case of Jupyter notebooks leave out the __file__ variable.
# AND ensure that the combination of .. leads to the root directory
project_root_path = os.path.realpath(os.path.join(__file__, "../../"))
sys.path.append(project_root_path)

import re
import matplotlib.pyplot as plt
import numpy as np

from src.utils.settings import Config


class _RegexSearch:

    def __init__(self):
        self.used_regex = None
        self.column = ''
        self.regex = ''
        self.flags = None
        self.index = None
        self.df = None


class DataExplorer:

    def __init__(self):
        self.search = _RegexSearch()

    def is_same_search(self, df, regex, column, use_regex, flags):
        return self.search.regex == regex \
            and self.search.column == column \
            and self.search.df is df \
            and self.search.used_regex == use_regex \
            and self.search.flags == flags

    def find_submissions(self, df, regex: str, column: str, use_regex=True, flags=re.IGNORECASE):
        if not self.is_same_search(df, regex, column, use_regex, flags):

            self.search.used_regex = use_regex
            self.search.column = column
            self.search.regex = regex
            self.search.flags = flags
            self.search.df = df

            if use_regex:
                pattern = re.compile(regex, flags=flags)
                idx = df[df[column].map(lambda s: re.search(pattern, s) is not None)].index
            else:
                pattern = None
                idx = df.index
            self.search.index = np.zeros_like(idx)
            self.search.index[:] = idx
            if Config.debug_mode:
                print(f"{len(self.search.index):,} appropriate submissions found")

    def display_random_submission(self, df):
        np.random.shuffle(self.search.index)

        idx = self.search.index[:1]
        submissions = zip(df["prompt"][idx], df["prompt_body"][idx], df["story"][idx])

        for prompt, prompt_body, story in submissions:
            sep = 79 * '~' + '\n'
            print(prompt)
            print(sep)

            if self.search.used_regex:
                mo = re.findall(self.search.regex, story)
                if mo is not None:
                    for match in mo:
                        print(match)
                    print(sep)

            print(prompt_body)
            print(sep)
            print(story)

    @staticmethod
    def plot_hist(values, range_: tuple, hist_type: str, value_src: str):
        left_border = range_[0]
        right_border = range_[1]
        bins = right_border - left_border
        while bins <= 100:
            bins *= 2
        while bins > 100:
            bins /= 2
        bins = int(round(bins + 1))
        title = f'{value_src} - {hist_type}'

        _ = plt.hist(values, range=range_, bins=bins, density=True)
        plt.axis([left_border, right_border, 0, None])
        plt.grid(True, linewidth=1.0)
        plt.xlabel(hist_type)
        plt.ylabel('Probability')
        _ = plt.title(title, fontdict={'fontsize': 18})
