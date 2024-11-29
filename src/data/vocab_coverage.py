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

import gc
import re
import itertools

from collections import Counter
from pathlib import Path
from tqdm import tqdm

from src.utils.settings import Config


class VocabCoverage:

    def __init__(self):
        self.l_tokenized = []
        self.d_vocab = {}

        # Contains tokens which are in the corpus vocabulary AND word embedding file
        self.d_tokens = {}

        # Alternative token splitting regular expressions
        # \w+[^\s]*\w+   \b[^\s]+?\b|\w+[^\s]*\w+|\w+   [^()\s]+|[()]   (\w+|[^\w\s])
        self.ptrn_token_split = re.compile(r'(\w+|[^\w\s])', re.IGNORECASE)

    def __load_words_not_vectors(self, file_path, vec_count, ft_format=False, encoding='latin'):
        """
        Loads tokens from a word vector file row wise with standard Python code

        Parameters
        ----------
        file_path: str
        vec_count: int
            Number of word vectors - rows without eventual header -
            in the file
        ft_format: bool
            True, if the word vector file is in the fastText format.
            False else.
        encoding: str
            String encoding of the word vector file
        """
        word_vec_file = Path(file_path)
        if not word_vec_file.is_file():
            raise IOError("File {:} does not exist! Please follow "
                          "Readme.md instructions".format(file_path))

        with open(file_path, mode='r', encoding=encoding) as file_:
            if ft_format:
                _ = file_.readline()

            for line in tqdm(file_, total=vec_count, unit=' vectors', unit_scale=True):
                pieces = line.split(' ')
                if pieces[0] in self.d_vocab:
                    self.d_tokens[pieces[0]] = True
                del pieces

    def __str_list_to_vocab(self, data):
        """
        The vocab of a list of docs

        Parameters
        ----------
        data : list-like
            Contains a sequence of strings. They will be transformed
        Returns
        -------
        list
            list of lists whose inner lists contain string tokens as elements
        """
        l_tokenized = data.apply(lambda s: re.findall(self.ptrn_token_split, s)).values
        return l_tokenized

    def __count_tokens(self, l_tokens):
        """
        Parameters
        ----------
        l_tokens: list
            A list which contains list of lists
            whose inner lists contain string tokens as elements
        Returns
        -------
        Counter
             dict of form {'<token>': <count>}
        """
        # Flatten the list of list of lists to a list
        fi_func = itertools.chain.from_iterable
        l_token_flat = list(fi_func(fi_func(l_tokens)))
        return Counter(l_token_flat)

    def __add_missing_words(self):
        count = 0
        for word in self.d_vocab:
            lower_case = (word == word.lower())
            word_lower = word.lower()
            found_upper = False if lower_case else word in self.d_tokens
            found_lower = word.lower() in self.d_tokens

            if not lower_case and (found_upper and not found_lower):
                self.d_tokens[word_lower] = self.d_tokens[word]
                count += 1
            elif not lower_case and (not found_upper and found_lower):
                self.d_tokens[word] = self.d_tokens[word_lower]
                count += 1
        if Config.debug_mode:
            print(f"Added {count} tokens to vocab")

    def __check_coverage(self):
        known_words = {}
        unknown_words = {}
        nb_known_words = 0
        nb_unknown_words = 0

        for word in self.d_vocab.keys():
            if word in self.d_tokens:
                known_words[word] = self.d_vocab[word]
                nb_known_words += self.d_vocab[word]
            else:
                unknown_words[word] = self.d_vocab[word]
                nb_unknown_words += self.d_vocab[word]

        covered_vocab = len(known_words) / len(self.d_vocab)
        covered_text = nb_known_words / (nb_known_words + nb_unknown_words)

        if Config.debug_mode:
            print('Found tokens for {:.2%} of d_vocab'.format(covered_vocab))
            print('Found tokens for {:.2%} of all text'.format(covered_text))
        unknown_words = sorted(unknown_words.items(), key=lambda kv: kv[1])[::-1]
        return unknown_words, covered_text, covered_vocab

    def calculate_oov(self, l_data, word_vector_path: str, vector_count: int
                      , ft_format=False, encoding='latin'):
        """
        Loads tokens from a word vector file row wise with standard Python code

        Parameters
        ----------
        l_data: list
            list containing strings of text
        word_vector_path: str
            Path to the word vector (word embeddings) file
        vector_count: int
            Number of word vectors - rows without eventual header -
            in the file
        ft_format: bool
            True, if the word vector file is in the fastText format.
            False else.
        encoding: str
            String encoding of the word vector file

        Returns
        -------
        list, float, float
            1. Out of vocabulary words as tuples in a list. First tuple element
             the token, second one the total count in the text corpus
            2. The percentage of covered text - how much text we can cover with
               our words/tokens in the vocabulary
            3. How many (percentage) word types in the text are in the vocabulary
        """
        for l_strings in l_data:
            self.l_tokenized.append(self.__str_list_to_vocab(l_strings))

        self.d_vocab = self.__count_tokens(self.l_tokenized)
        del self.l_tokenized

        self.__load_words_not_vectors(str(word_vector_path)
                                      , vec_count=vector_count
                                      , ft_format=ft_format
                                      , encoding=encoding)
        self.__add_missing_words()
        oov_tokens, _, _ = self.__check_coverage()
        del self.d_vocab
        del self.d_tokens
        gc.collect()

        # To make a rerun of this function possible without
        # creating a new VocabCoverage object
        self.l_tokenized = []
        self.d_vocab = {}
        self.d_tokens = {}
        return oov_tokens
