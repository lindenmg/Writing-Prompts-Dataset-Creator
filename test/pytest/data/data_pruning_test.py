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
from pyfasttext import FastText

from src.data.data_pruning import DataPruner
from src.utils.settings import Config

path_test_data = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


def load_data(file: str):
    path_file = Path(path_test_data) / file

    if not path_file.is_file():
        raise FileNotFoundError(f"{str(path_file)}")

    return pd.read_csv(path_file, keep_default_na=False)


class TestDataPruner:

    @staticmethod
    def check_pruning(df, function):
        for _, row in df.iterrows():
            pruned = function(row['raw_text'])

            # print()
            # print("TEST:" + str(pruned) + "|||")
            # print("GOLD:" + str(row['clean_text']) + "|||")
            # print()

            assert pruned == row['clean_text']

    @staticmethod
    def check_series(test, gold):
        for t, g in zip(test, gold):
            print()
            print("TEST:" + str(t) + "|||")
            print("GOLD:" + str(g) + "|||")
            print()
            assert t == g

    def test_flag_for_removal(self):
        pruner = DataPruner()
        df = load_data('prune_nsfw_and_removed_test.csv')

        self.check_pruning(df, pruner.flag_for_removal)

    def test_prune_prompt_body(self):
        pruner = DataPruner()
        df = load_data('prune_prompt_body_test.csv')

        self.check_pruning(df, pruner.prune_prompt_body)

    def test_prune_submission(self):
        pruner = DataPruner()
        df = load_data('prune_submission_test.csv')

        self.check_pruning(df, pruner.prune_submission)

    def test_prune_story(self):
        pruner = DataPruner()
        df = load_data('prune_story_test.csv')

        self.check_pruning(df, pruner.prune_story_text)

    def test_detect_lang_ext(self):
        path_lang = Path(Config.path.data_folder) / 'external' / 'lid.176.bin'
        ft_model = FastText(str(path_lang))
        pruner = DataPruner()
        df = load_data('prune_language_test.csv')
        pruner.ft_predict = ft_model.predict_proba_single

        self.check_pruning(df, pruner.detect_lang_ext)
        del pruner.ft_predict

    @pytest.mark.parametrize("cpu_cores", [1, 2, 4])
    def test_prune_data_i(self, cpu_cores):
        pruner = DataPruner()
        df_clean = load_data('prune_data_clean_test.csv')
        df_raw = load_data('prune_data_raw_test.csv')

        pruner.prune_data(df_raw, cpu_cores=cpu_cores)

        self.check_series(df_raw['prompt'], df_clean['prompt'])
        self.check_series(df_raw['prompt_body'], df_clean['prompt_body'])
        self.check_series(df_raw['story'], df_clean['story'])

    def test_prune_data_ii(self):
        pruner = DataPruner()
        df = load_data('prune_story_test.csv')

        with pytest.raises(ValueError):
            pruner.prune_data(df)
