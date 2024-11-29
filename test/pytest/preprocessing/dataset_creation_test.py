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
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.dataset_creation import DataSetCreator, retokenize_bert, retokenize_bpe_str_sent_list
from src.utils.helpers import parallel_pandas
from src.utils.settings import Config

path_test_data = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


def load_data(file: str):
    path_file = Path(path_test_data) / file

    if not path_file.is_file():
        raise FileNotFoundError(f"{str(path_file)}")

    return pd.read_csv(path_file, keep_default_na=False)


def load_df():
    data_base = Config.path.data_base
    sql_query = """
    SELECT f.prompt, f.prompt_body, f.story, f.prompt_score, f.story_score, f.story_words
    FROM filtered as f
    WHERE f.story_words >= 100 and f.story_words <= 850
    order by f.prompt ASC, f.story_score DESC, f.prompt_score DESC;"""
    conn = sqlite3.connect(data_base)
    return pd.read_sql_query(sql_query, conn)


def story_for_bert(story):
    story = re.sub(ptrn_space_hull, r' \1 ', story)
    story = story.replace('_', ' _ ')
    story = re.sub(r'\s+', ' ', story)
    story = re.sub(r'^ | $', '', story)
    return story.replace('\n', '')


ptrn_space_hull = re.compile(r'([^\s\w©])')
cpu_cores = Config.hardware.n_cpu


class TestDataSetCreator:

    @staticmethod
    def check_filtering(df, function):
        for _, row in df.iterrows():
            filtered = function(row['raw_text'])

            # print()
            # print("DONE    :" + str(filtered) + "|||")
            # print("FILTERED:" + str(row['clean_text']) + "|||")
            # print()

            assert filtered == row['processed_text']

    @staticmethod
    def check_text_with_ukn(o, c, ukn_token):
        equal = True
        old = 0

        for match in re.finditer(ukn_token, c):
            new = match.span()
            if old != new[0]:
                good_text = c[old[1]:new[0]]
                if good_text not in o:
                    equal = False
                    break
            old = new
        return equal

    def check_series(self, test, gold, ukn_token):
        l_is_equal = []

        for i, (t, g) in enumerate(zip(test, gold)):
            are_equal = (t == g)
            if ukn_token in t and not are_equal:
                are_equal = self.check_text_with_ukn(t, g, ukn_token)
            if not are_equal:
                print()
                print(f'idx: {i}')
                print("Processed:" + str(t) + "|||")
                print(79 * '~')
                print("Original :" + str(g) + "|||")
                print()
                print(79 * '=')
            l_is_equal.append(are_equal)
        assert (np.asarray(l_is_equal) is True).all()

    def test_group_prompts(self):
        creator = DataSetCreator(vocab_name='bpe_710w_1024t_28k')
        df = load_df()[0:None]

        df = creator.group_prompts(df)
        idx_max = df['prompt_idx'][len(df) - 1] + 1
        len_unique = len(df.drop_duplicates(subset='prompt_idx'))
        assert len_unique == idx_max

    @pytest.mark.parametrize("joint_vocab", [False, True])
    def test_tokenize_story_bpe(self, joint_vocab):
        creator = DataSetCreator(vocab_name='bpe_unit_test_vocab'
                                 , vocab_size=8192
                                 , use_joint_vocab=joint_vocab)
        df = load_df()[0:1_000]

        df = creator.group_prompts(df)
        creator.split_set(df)
        creator._train_spm_model(df)
        creator.tokenize_own_vocab(df, cpu_cores)
        parallel_pandas(df=df
                        , func=retokenize_bpe_str_sent_list
                        , column='story_bpe_tokens'
                        , new_column='story_retokenized_bpe'
                        , num_threads=cpu_cores)

        # CAUTION !!! There are tokenization errors due to the fact that
        # it splits sentences where a space in between is missing
        # Those cases are almost always handled correctly.
        # Exception: "I'm talking."I said ==> "I'm talking. "I said
        # No other errors known
        # Okay as long as only ids: [307, 414, 555, 611, 729, 749, 838] fail
        self.check_series(df['story_retokenized_bpe']
                          , df['story']
                          , ukn_token=' ⁇ ')

    def test_tokenize_story_bert(self):
        creator = DataSetCreator(vocab_name='bpe_710w_1024t_28k')
        df = load_df()[0:1_000]

        creator.tokenize(df, 'story', cpu_cores)
        parallel_pandas(df=df
                        , func=retokenize_bert
                        , column='story_bert_tokens'
                        , new_column='story_retokenized_bert'
                        , num_threads=cpu_cores)
        parallel_pandas(df=df
                        , func=story_for_bert
                        , column='story'
                        , new_column='story_bert'
                        , num_threads=cpu_cores)
        self.check_series(df['story_retokenized_bert']
                          , df['story_bert']
                          , ukn_token=' ⁇ ')

    def test_tokenize_prompts(self):
        creator = DataSetCreator(vocab_name='bpe_710w_1024t_28k')
        df = load_df()[0:None]

        creator.tokenize(df, 'prompt', cpu_cores)
        parallel_pandas(df=df
                        , func=retokenize_bert
                        , column='prompt_bert_tokens'
                        , new_column='prompt_retokenized_bert'
                        , num_threads=cpu_cores)
        self.check_series(df['prompt_retokenized_bert'], df['prompt'])

    def test_split_set(self):
        creator = DataSetCreator(vocab_name='bpe_710w_1024t_28k')
        df = load_df()[0:None]
        df = creator.group_prompts(df)
        creator.split_set(df)

        len_src = len(df)
        train = df[df['val'] == 0]
        test = df[df['test']]
        val = df[df['val'] == 1]
        len_train = len(train)
        len_test = len(test)
        len_val = len(val)

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)

        train.drop_duplicates(subset='prompt_idx', inplace=True)
        test.drop_duplicates(subset='prompt_idx', inplace=True)
        val.drop_duplicates(subset='prompt_idx', inplace=True)
        len_ideal = len(train) + len(test) + len(val)
        df = pd.concat([test, val, train])
        df.drop_duplicates(subset='prompt_idx', inplace=True)
        len_final = len(df)
        len_oof = 0.1 * len_src
        assert len_ideal == len_final
        assert 0.999 * len_oof <= len_test <= 1.001 * len_oof
        assert 0.999 * len_oof <= len_val <= 1.001 * len_oof
        assert 0.998 * 0.8 * len_src <= len_train <= 1.002 * 0.8 * len_src
        assert len_ideal < len_src
