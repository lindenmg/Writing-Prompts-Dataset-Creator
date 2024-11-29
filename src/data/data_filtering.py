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
import textstat

from collections import Counter

from src.data.spacyoverlay import SpacyOverlay
from src.utils.helpers import parallel_pandas, section_text
from src.utils.settings import Config


class DataFilterer:

    def __init__(self, ):
        """
        Removes data-points in the tail of statistical distributions

        Story filtering:

        - Drop low word lengths to limit repetitive text and poems.
        - Drop high word lengths to limit strange structured texts
        - Drop texts with to frequent tokens to limit repetitive content.
        - Drop texts with to infrequent tokens to limit poems & weirdness
        - Drop texts which have a too high or low ratio of special characters
          to limit weird or no punctuation & poems
        - Drop texts with to many or few different characters to limit texts
          with binary & morse code and foreign languages in it
        - Drop texts with to long sentences for regularization & fewer poems
        - Drop texts with to low readability score for regularization
        - Drop texts with to high readability to get rid of poems & weirdness

        Prompt filtering:

        - Drop texts with only one word (whitespaces define word-boundaries)
          Prompts have to have a tag (e.g. [WP]) so one word is no prompt
        - Prompts with fewer than 8 characters, because that doesn't make
          a suitable prompt.

        Prompt filtering is quite limited because the NLG system has to be
        able to handle as many prompts as possible. Also, the human moderators
        of the data source have a good overview of the prompt title texts -
        so to weird prompts have been deleted by them.
        Contrary to that story texts are relatively heavily normalized to make
        it easier for the probabilistic NLG system. Although in almost all
        cased only the long end of the distribution tail is cut off.
        Which still leaves a shorter tail to let the NLG system learn some
        irregularities.
        """
        self.ptrn_whitespace = re.compile(r'\s')
        self.ptrn_tokenize = re.compile(r'(\w+|[^\w\s])')
        self.ptrn_sc = re.compile(r'[^\w\s]')
        self.ptrn_an = re.compile(r'\w')

        spacy = SpacyOverlay()
        self.nlp = spacy.get_nlp()  # ToDo: Replace Spacy by NLTK sentence piece model to split sentences uniformly

    @staticmethod
    def _drop_data(df, idx):
        df.drop(idx, axis='index', inplace=True)
        del idx

    def token_freq(self, text):
        tokens = re.findall(self.ptrn_tokenize, text)
        most_freq = Counter(tokens).most_common(1)[0][1]
        return most_freq / len(tokens)

    def sent_len_max(self, text):
        doc = self.nlp(text)
        max_len = 0
        for sent in doc.sents:
            sent_len = len(sent)
            if sent_len > max_len:
                max_len = sent_len
        return max_len

    def sent_len_avg(self, text):
        doc = self.nlp(text)
        tokens = len(doc)
        sentences = 0
        for _ in doc.sents:
            sentences += 1
        return tokens / sentences

    def _drop_rows_conditionally(self, df, column: str
                                 , high_bound=None
                                 , low_bound=None
                                 , quantile=True
                                 , cli_text='Computing...'):
        if high_bound is not None and low_bound is not None \
                and high_bound <= low_bound:
            raise ValueError(f'low_bound >= high_bound: '
                             f'{low_bound} >= {high_bound} !')
        if column not in df.columns:
            raise ValueError(f'column {column} not in df.columns!')

        if Config.debug_mode:
            section_text(cli_text)

        if high_bound is not None:
            if quantile:
                high_bound = df[column].quantile(q=high_bound)
            idx_high = df[df[column] > high_bound].index
            self._drop_data(df, idx_high)

        if low_bound is not None:
            if quantile:
                low_bound = df[column].quantile(q=low_bound)
            idx_low = df[df[column] < low_bound].index
            self._drop_data(df, idx_low)

    def _calc_stats_i(self, df, cpu_cores):
        if Config.debug_mode:
            section_text("Compute statistical information for filtering")

        # Story word & chars stats
        df['story_words'] = df.story.apply(
            lambda s: len(re.split(self.ptrn_whitespace, s))
        )
        df['story_chars'] = df.story.str.len()
        df['story_word_len'] = df['story_chars'] / df['story_words']

        df['story_special_chars'] = df.story.apply(
            lambda s: len(re.findall(self.ptrn_sc, s))
        )

        df['story_alpha_nums'] = df.story.apply(
            lambda s: len(re.findall(self.ptrn_an, s))
        )
        df['story_sc_by_an'] = df.story_special_chars / df.story_alpha_nums

        df['story_unique_chars'] = df.story.apply(lambda s: len(set(s)))

        parallel_pandas(func=self.token_freq
                        , df=df
                        , column='story'
                        , new_column='story_token_freq'
                        , num_threads=cpu_cores)

        # Prompt word & chars stats
        df['prompt_words'] = df.prompt.apply(
            lambda s: len(re.split(self.ptrn_whitespace, s))
        )
        df['prompt_chars'] = df['prompt'].str.len()

    def _drop_stories_i(self, df):
        self._drop_rows_conditionally(df=df, column='story_chars'
                                      , high_bound=10_000
                                      , low_bound=None
                                      , quantile=False
                                      , cli_text="Drop stories with more than "
                                                "10,000 characters")

        self._drop_rows_conditionally(df=df, column='story_word_len'
                                      , high_bound=0.99
                                      , low_bound=0.01
                                      , quantile=True
                                      , cli_text="Keep stories with 1% <= "
                                                "word-length-quantile <= 99%")

        self._drop_rows_conditionally(df=df, column='story_sc_by_an'
                                      , high_bound=0.99
                                      , low_bound=0.005
                                      , quantile=True
                                      , cli_text="Keep stories with 0.5% <= "
                                                "special-chars / alpha-num-"
                                                "chars <= 99%")

        self._drop_rows_conditionally(df=df, column='story_unique_chars'
                                      , high_bound=0.998
                                      , low_bound=0.002
                                      , quantile=True
                                      , cli_text="Keep stories with 0.2% <= "
                                                "unique-chars <= 99.8%")

        self._drop_rows_conditionally(df=df, column='story_token_freq'
                                      , high_bound=0.99
                                      , low_bound=0.001
                                      , quantile=True
                                      , cli_text="Keep stories with 0.1% <= "
                                                "most-freq-token / "
                                                "total_tokens <= 99%")

    def _calc_stats_ii(self, df, cpu_cores):
        if Config.debug_mode:
            section_text("Compute more statistical information for filtering")

        # Sentence & reading difficulty stats
        parallel_pandas(func=self.sent_len_avg
                        , df=df
                        , column='story'
                        , new_column='story_sent_len_avg'
                        , num_threads=cpu_cores)

        parallel_pandas(func=self.sent_len_max
                        , df=df
                        , column='story'
                        , new_column='story_sent_len_max'
                        , num_threads=cpu_cores)

        parallel_pandas(func=textstat.flesch_reading_ease
                        , df=df
                        , column='story'
                        , new_column='story_fres'
                        , num_threads=cpu_cores)

    def _drop_stories_ii(self, df):
        self._drop_rows_conditionally(df=df, column='story_sent_len_max'
                                      , high_bound=0.98
                                      , low_bound=None
                                      , quantile=True
                                      , cli_text="Keep stories with max-"
                                                "sentence-length <= 98%")

        self._drop_rows_conditionally(df=df, column='story_sent_len_avg'
                                      , high_bound=0.99
                                      , low_bound=None
                                      , quantile=True
                                      , cli_text="Keep stories with avg-"
                                                "sentence-length <= 99%")

        self._drop_rows_conditionally(df=df, column='story_fres'
                                      , high_bound=0.999
                                      , low_bound=None
                                      , quantile=True
                                      , cli_text="Keep stories with 0 <= Flesch"
                                                 " reading ease score <= 99.9")

        idx_low = df[df['story_fres'] < 0].index
        self._drop_data(df, idx_low)

    def _drop_prompts(self, df):
        if Config.debug_mode:
            section_text("Drop prompts with less than "
                         "two words and seven chars")

        idx_words = df[df['prompt_words'] < 2].index
        self._drop_data(df, idx_words)

        idx_chars = df[df['prompt_chars'] < 8].index
        self._drop_data(df, idx_chars)

    def filter_data(self, df, cpu_cores: int):
        """
        Removes data-points in the tail of statistical distributions

        Parameters
        ----------
        df: pandas.DataFrame
        cpu_cores: int
        """
        if 'story' not in df.columns \
                or 'prompt_body' not in df.columns \
                or 'prompt' not in df.columns:
            raise ValueError(f"``df`` should contain at least string columns "
                             f"'story', 'prompt' & 'prompt_body', but does "
                             f"only contain {df.columns}")

        # ==== CALC STATS ====================================================

        self._calc_stats_i(df=df, cpu_cores=cpu_cores)

        # ==== DROP STORIES ==================================================

        self._drop_stories_i(df=df)

        # ==== CALC STATS ====================================================

        # We have to split the statistical information computation because
        # there are some binary & Morse code sequences which SpaCy can't
        # split into sentences and therefore throws an error.
        # But they get conveniently filtered out by previous steps
        self._calc_stats_ii(df=df, cpu_cores=cpu_cores)

        # ==== DROP STORIES ==================================================

        self._drop_stories_ii(df=df)

        # ==== DROP PROMPTS ==================================================

        self._drop_prompts(df=df)
