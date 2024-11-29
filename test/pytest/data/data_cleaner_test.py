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
import pytest

from src.data.data_cleaning import DataCleaner

path_test_data = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


def load_data(file: str):
    path_file = Path(path_test_data) / file

    if not path_file.is_file():
        raise FileNotFoundError(f"{str(path_file)}")

    return pd.read_csv(path_file)


def split_dataframe(df):
    df_new = pd.DataFrame()
    df_new['prompt_body'] = df[2:4]['raw_text'].reset_index(drop=True)
    df_new['prompt'] = df[4:6]['raw_text'].reset_index(drop=True)
    df_new['story'] = df[0:2]['raw_text'].reset_index(drop=True)

    df_new['pb_clean'] = df[2:4]['clean_text'].reset_index(drop=True)
    df_new['p_clean'] = df[4:6]['clean_text'].reset_index(drop=True)
    df_new['s_clean'] = df[0:2]['clean_text'].reset_index(drop=True)
    return df_new


class TestDataCleaner:

    @staticmethod
    def check_cleaning(df, function, function_pre=None):
        for _, row in df.iterrows():
            if function_pre is not None:
                cleaned = function_pre(row['raw_text'])
                cleaned = function(cleaned)
            else:
                cleaned = function(row['raw_text'])

            # print()
            # print("DONE :" + cleaned + "|||")
            # print("CLEAN:" + row['clean_text'] + "|||")
            # print()

            assert cleaned == row['clean_text']

    def test_clean_linebreaks(self):
        cleaner = DataCleaner()
        df = load_data('clean_linebreaks_test.csv')

        self.check_cleaning(df, cleaner.clean_linebreaks, cleaner.clean_spaces)

    def test_clean_spaces(self):
        cleaner = DataCleaner()
        df = load_data('clean_spaces_test.csv')

        self.check_cleaning(df, cleaner.clean_spaces)

    def test_clean_prompt(self):
        cleaner = DataCleaner()
        df = load_data('clean_prompt_test.csv')

        self.check_cleaning(df, cleaner.clean_prompt)

    def test_clean_text(self):
        cleaner = DataCleaner()
        df = load_data('clean_text_test.csv')

        self.check_cleaning(df, cleaner.clean_text)

    @pytest.mark.parametrize("cpu_cores", [1, 2, 4])
    def test_clean_data_i(self, cpu_cores):
        cleaner = DataCleaner()
        df = load_data('clean_data_test.csv')
        df = split_dataframe(df)

        cleaner.clean_data(df, cpu_cores=cpu_cores)

        for _, row in df.iterrows():
            assert row['prompt_body'] == row['pb_clean']
            assert row['prompt'] == row['p_clean']
            assert row['story'] == row['s_clean']

    def test_clean_data_ii(self):
        cleaner = DataCleaner()
        df = load_data('clean_data_test.csv')

        with pytest.raises(ValueError):
            cleaner.clean_data(df)
