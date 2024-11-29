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
project_root_path = os.path.realpath(os.path.join("../"))
sys.path.append(project_root_path)

import itertools
import math
import nltk.data
import numpy as np
import pandas as pd
import re
import sentencepiece as spm
import torch

from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import parallel_pandas, section_text, round_to_next_multiple
from src.preprocessing.token_encoding import encode_pieces
from src.utils.settings import Config


# ToDo: Unify dataset creations
#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def retokenize_bert(ids):
    ids = [sent.split() for sent in ids]
    ids = list(itertools.chain.from_iterable(ids))
    ids = list(map(int, ids))
    tokens = tokenizer_pretrained.convert_ids_to_tokens(ids, skip_special_tokens=True)
    tokens = tokenizer_pretrained.convert_tokens_to_string(tokens)
    return tokens


def retokenize_bpe_str_sent_list(l_ids):
    l_ids = [sent.split() for sent in l_ids]
    l_ids = list(itertools.chain.from_iterable(l_ids))
    l_ids = list(map(int, l_ids))
    return retokenize_bpe_int_list(l_ids)


def retokenize_bpe_int_list(l_ids):
    pieces = spp.decode_ids(l_ids)
    pieces = "".join(pieces)
    pieces = pieces.replace('<newline>', '\n')

    # Remove whitespace after linefeed except when it is an unknown token
    pieces = re.sub(ptrn_space_start, r'\n\1', pieces)
    return pieces


def load_spp_bpe_vocab(vocab_name: str):
    path_model = Path(Config.path.project_root_folder) / "models" / f"{vocab_name}.model"
    spp.Load(str(path_model))


def create_tokens_array(tokens):
    l_tokens = tokens.replace('|▁|', '').split()
    l_tokens = list(map(int, l_tokens))
    return l_tokens


def create_torch_token_dataset(pd_series, len_story=512, max_stories=None):
    l_tokens = pd_series.apply(create_tokens_array).values
    len_ = min(max_stories, len(l_tokens))
    s = set()

    if Config.debug_mode:
        print(f"Create ({len_:,d} | {len_story}) tensor for dataset")

    t_tokens = torch.LongTensor(len_, len_story).zero_()
    t_length = torch.LongTensor(len_).zero_()
    for i in range(len_):
        end = len(l_tokens[i])
        s = s.union(set(l_tokens[i]))
        t_length[i] = end
        t_tokens[i, :end] = torch.LongTensor(l_tokens[i])
    print(f"{len(s):,d} individual tokens in data-subset")
    return t_tokens, t_length


def get_prompt_data(df, df_prompts, prompt_size, limit_stories=math.inf):
    df_data = pd.merge(df, df_prompts, left_on='prompt_idx', right_index=True, how='right')
    t_prompts_train = df_data[df_data.val == 0]
    t_prompts_train.reset_index(drop=True, inplace=True)

    # Select prompt feature vector columns with np.arrange part. ToDo: vector dim parameter
    t_story_idx_train = t_prompts_train.loc[0:limit_stories - 1, 'story_idx']
    t_prompts_train = t_prompts_train.loc[0:limit_stories - 1, np.arange(0, prompt_size)]
    t_prompts_val = df_data[df_data.val == 1]
    t_prompts_val.reset_index(drop=True, inplace=True)
    t_story_idx_val = t_prompts_val.loc[0:math.inf, 'story_idx']
    t_prompts_val = t_prompts_val.loc[0:math.inf, np.arange(0, prompt_size)]
    print(f"train.shape: {t_prompts_train.shape}; val.shape: {t_prompts_val.shape}")

    t_story_idx_train = torch.from_numpy(t_story_idx_train.values).contiguous()
    t_prompts_train = torch.from_numpy(t_prompts_train.values).contiguous()
    t_story_idx_val = torch.from_numpy(t_story_idx_val.values).contiguous()
    t_prompts_val = torch.from_numpy(t_prompts_val.values).contiguous()
    return (t_prompts_val, t_prompts_train), (t_story_idx_val, t_story_idx_train)


# !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!
# We have to place these functions & variable initialization outside
# DatasetCreator because otherwise the parallel_pandas function - needed for
# faster tokenization - won't work. This is because the multiprocessing.Pool.map
# function relies on pickling. And there is a weird behaviour with class
# functions in these case. So this here is a workaround.
# !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!   !!!

sentence_detector = nltk.data.load(Config.path.nltk_sentence_tokenizer)
ptrn_paragraph = re.compile(r'\n\n.+|\n.+|.+')
ptrn_space_start = re.compile(r'\n ([^⁇])')
spp = spm.SentencePieceProcessor()

# For sentences were special chars at the end have been split
# into the next sentence
splitted_ptrn = re.compile(r'^([^\w\s\-]+ |\n+)(.+)', flags=re.IGNORECASE)
broken_ptrn = re.compile(r'^(\W+)$')
tokenizer_pretrained = AutoTokenizer.from_pretrained(Config.path.pretrained_model,
                                                     cache_dir=Config.path.model_folder,
                                                     do_lower_case=False)


def encode_ids_prompt_bert(text):
    ids = tokenizer_pretrained.encode(text, add_special_tokens=True)
    ids = map(str, ids)
    return " ".join(ids)


def split_sentences(text):
    # We split by linebreaks to help the sentence split model
    # Due to the formatting, sentences don't extend over several lines
    paragraphs = re.findall(ptrn_paragraph, text)
    l_new_sents = []
    l_sents = sentence_detector.tokenize_sents(paragraphs)
    l_sents = list(itertools.chain.from_iterable(l_sents))
    l_new_sents.append(l_sents[0])
    last_i = 0

    # The sentence splitter tends to split sentences with unusual special chars
    # at the end into two sentences. Therefore, we combine them again
    for i, sent in enumerate(l_sents[1:], start=1):
        match = re.match(splitted_ptrn, sent)
        if match is not None:
            l_new_sents[last_i] += match[1][:-1] if match[1][-1] == ' ' else match[1]
            l_new_sents.append(match[2])
            last_i += 1
        else:
            match = re.match(broken_ptrn, sent)

            # Sometimes e.g. 'end of sentence! :)'
            # gets split into 'end of sentence!' & ':)'
            space_split = l_new_sents[last_i] + ' ' + sent
            if space_split in text:
                sent = ' ' + sent
            if match is not None:
                l_new_sents[last_i] += sent
            else:
                l_new_sents.append(sent)
                last_i += 1
    return l_new_sents


def encode_ids_sents_bert(text):
    sents = split_sentences(text)

    for i, sent in enumerate(sents):
        sents[i] = tokenizer_pretrained.encode(sent, add_special_tokens=True)
        sents[i] = map(str, sents[i])
    return [" ".join(sent) for sent in sents]


def encode_ids_prompt_bpe(text):
    text = encode_pieces(spp, text, return_unicode=False, sample=False)
    text = ['<s>'] + text + ['</s>']
    ids = [spp.PieceToId(piece) for piece in text]
    ids = " ".join(map(str, ids))
    return ids


def encode_ids_sents_bpe(text):
    sents = split_sentences(text)

    for i, sent in enumerate(sents):
        sent = sent.replace('\n', '<newline>')
        sents[i] = encode_pieces(spp, sent, return_unicode=False, sample=False)

    sents[-1].append('</s>')
    sents[0] = ['<s>'] + sents[0]
    ids = [
        [spp.PieceToId(piece) for piece in sent]
        for sent in sents
    ]
    ids = [" ".join(map(str, sent_ids)) for sent_ids in ids]
    return ids


#  ____        _        ____       _    ____                _
# |  _ \  __ _| |_ __ _/ ___|  ___| |_ / ___|_ __ ___  __ _| |_ ___  _ __
# | | | |/ _` | __/ _` \___ \ / _ \ __| |   | '__/ _ \/ _` | __/ _ \| '__|
# | |_| | (_| | || (_| |___) |  __/ |_| |___| | |  __/ (_| | || (_) | |
# |____/ \__,_|\__\__,_|____/ \___|\__|\____|_|  \___|\__,_|\__\___/|_|

class DataSetCreator:

    def __init__(self, vocab_name: str, vocab_size: int = 24576
                 , use_joint_vocab: bool = False, tokenization='unigram'):
        """
        For tokenization and splitting of the data into train, val & test sets

        Parameters
        ----------
        vocab_name: str
            Name of the vocab for database hash name
        vocab_size: int
            The amount of tokens in the vocabulary
        use_joint_vocab: bool
            If `True` then prompt and story text from the training set will
            be used to train a joint vocabulary. E.g. for own sentence embeddings
            of the prompt+story context.
            If `False` then only the story gets used for training the tokenizer model
        tokenization: str
            'unigram' or 'bpe' for the respective low-level tokenization techniques
            of the SentencePiece library
        """
        np.random.seed(1337)
        self.tokenization_model = tokenization
        self.use_joint_vocab = use_joint_vocab
        self.vocab_name = vocab_name
        self.vocab_size = vocab_size

    def tokenize(self, df, column, cpu_cores):
        if column == 'story':
            encode_func_b = encode_ids_sents_bert
        elif column == 'prompt':
            encode_func_b = encode_ids_prompt_bert
        else:
            raise ValueError(f"Use 'story' or 'prompt' as tokenization "
                             f"goal column, not {column}")

        new_column = column + "_bert_tokens"
        parallel_pandas(func=encode_func_b
                        , df=df
                        , column=column
                        , new_column=new_column
                        , num_threads=cpu_cores)

    def _train_spm_model(self, df):
        """
        Trains a sentence piece module which tokenizes text into word pieces.

        Parameters
        ----------
        df: pandas.DataFrame

        See Also
        -------
        https://github.com/google/sentencepiece
        """
        path_text = Path(Config.path.data_folder) / "external" / "story_text.txt"
        path_model = Path(Config.path.project_root_folder) / "models" / self.vocab_name

        # Character coverage isn't 1.0 because we still have a lot of Smileys,
        # and some rare characters (russian...)
        options = f"""--input={str(path_text)}
                      --model_prefix={str(path_model)}
                      --vocab_size={self.vocab_size}
                      --character_coverage=0.9998
                      --model_type={self.tokenization_model}
                      --normalization_rule_name=identity
                      --control_symbols=<sep>,<Ctrl1>,<Ctrl2>
                      --user_defined_symbols=<newline>,<URL>,[WP],[RF],[SP],[TT]
                      --pad_id=0
                      --bos_id=1
                      --eos_id=2
                      --unk_id=3"""
        options = re.sub(r'\s+', ' ', options)

        # Train BPE vocab model on training data:
        with open(str(path_text), "w") as f:
            for story in df.story[df.val == 0]:
                for paragraph in story.split('\n'):
                    if len(paragraph) > 0:
                        f.write(paragraph + '\n')
            if self.use_joint_vocab:
                for prompt in df.prompt[df.val == 0]:
                    f.write(prompt + '\n')
        spm.SentencePieceTrainer.Train(options)

    def tokenize_own_vocab(self, df, cpu_cores):
        load_spp_bpe_vocab(self.vocab_name)
        parallel_pandas(func=encode_ids_sents_bpe
                        , df=df
                        , column="story"
                        , new_column="story_bpe_tokens"
                        , num_threads=cpu_cores)
        if self.use_joint_vocab:
            parallel_pandas(func=encode_ids_prompt_bpe
                            , df=df
                            , column="prompt"
                            , new_column="prompt_bpe_tokens"
                            , num_threads=cpu_cores)

    @staticmethod
    def split_set(df):
        np.random.seed(1337)
        test_len = int(round(len(df) * 0.1))
        val_len = test_len

        # We have to modify the validation- and tests-sets lengths a bit,
        # so we split at unique prompt boundaries.
        # This way submissions with the same prompt remain in one set
        while df['prompt_idx'][test_len + 1] == df['prompt_idx'][test_len]:
            test_len -= 1
        val_start = test_len + 1
        val_end = val_start + val_len

        while df['prompt_idx'][val_end + 1] == df['prompt_idx'][val_end]:
            val_end -= 1

        df['test'] = False
        df['val'] = 0

        # The end of the indexing slice is not exclusive!
        # So it is [0, test_len], NOT [0, test_len]
        df.loc[0:test_len, 'test'] = True
        df.loc[val_start:val_end, 'val'] = 1
        df.loc[0:test_len, 'val'] = -1

    @staticmethod
    def group_prompts(df):
        prompt_groups = [group for _, group in df.groupby('prompt', sort=False)]
        np.random.shuffle(prompt_groups)
        df_grouped = pd.concat(prompt_groups)
        df_grouped['prompt_idx'] = df_grouped.groupby('prompt', sort=False).ngroup()
        df_grouped['story_idx'] = np.arange(len(df_grouped))
        df_grouped.reset_index(drop=True, inplace=True)
        return df_grouped

    @staticmethod
    def calc_token_meta_data(df):
        # We do the counting for stories based on BPE tokens, because that
        # are those we use for the actual language generation
        df['story_sent_num'] = df.story_bpe_tokens.apply(lambda s: len(s))
        df['story_sent_len_max'] = df.story_bert_tokens.apply(lambda story:
                                                              max([len(sent.split())
                                                                   for sent in story])
                                                              )

        df['story_bert_tokens'] = df.story_bert_tokens.apply(lambda s: " |▁| ".join(s))
        df['story_bpe_tokens'] = df.story_bpe_tokens.apply(lambda s: " |▁| ".join(s))

        df['story_token_num'] = df.story_bpe_tokens.apply(lambda s: len(s.split()))
        df['prompt_token_num'] = df.prompt_bert_tokens.apply(lambda s: len(s.split()))
        df['story_token_num'] = df.story_token_num - df.story_sent_num + 1

        if Config.debug_mode:
            token_ratio = df.story_token_num / df.story_words
            print(f"Mean of token/words ratio: {token_ratio.mean():.3}")

    @staticmethod
    def filter_submissions(df, token_limit):
        """Filter regarding token length and sentence amount"""

        def drop_rows(column, value):
            idx = df[column > value].index
            df.drop(idx, axis='index', inplace=True)
            df.reset_index(drop=True, inplace=True)
            del idx

        qht = df.story_token_num.quantile(0.998)  # 1260.0 for 850 words
        qhs = df.story_sent_num.quantile(0.998)  # 106.0 for 850 words
        qhl = df.story_sent_len_max.quantile(0.998)  # 92 for 350 words & BPE tokens

        # Multiple of 8 due to potential use of TensorCores for the DNN training
        # -3 / + 5 to make room for three additional inputs (prompt, separator, eos)
        # and still be conforming with multiple of 8 scheme
        sentence_limit = int(round_to_next_multiple(qhs - 3, 8)) + 5
        sent_len_limit = int(round_to_next_multiple(qhl, 8))

        # the + 1 is for the last prediction (ground truth)
        token_limit += 1

        if Config.debug_mode:
            print(f"99.8 percentile of story token count:    {int(qht)}")
            print(f"99.8 percentile of story sentence count: {int(qhs)}")
            print(f"99.8 percentile of max sentence length:  {int(qhl)}")
            print(f"But we will use {token_limit}, {sentence_limit} & {sent_len_limit}")

        # If we use a different word count and are in case of the token
        # max story length to far off of our set limit, raise a Warning:
        if abs(qht - token_limit) / qht > 0.03:
            raise Warning("The token length boundary is to far off from the "
                          "quantile. Please check the (changed) text"
                          "data (distribution)!   <===   !!!")
        if qhl > sent_len_limit or qhs > sentence_limit:
            raise Warning("You are dropping to many stories due "
                          "to too harsh sentence (length) limits!")

        drop_rows(df.story_token_num, token_limit)
        drop_rows(df.story_sent_num, sentence_limit)
        drop_rows(df.story_sent_len_max, sent_len_limit)

    def create_dataset(self, df, cpu_cores, token_limit=512):

        if token_limit % 256 != 0:
            raise ValueError(f"`token_limit` should be a multiple of 256 due to the "
                             f"block-size for the DNN input, but is {token_limit}!")

        if Config.debug_mode:
            section_text("Grouping same prompts together...")
        df = self.group_prompts(df)

        if Config.debug_mode:
            section_text('Tokenize stories with BERT vocab I/II...')
        self.tokenize(df, column='story', cpu_cores=cpu_cores)

        if Config.debug_mode:
            section_text('Tokenize prompts with BERT vocab I/II...')
        self.tokenize(df, column='prompt', cpu_cores=cpu_cores)

        if Config.debug_mode:
            section_text('Split data into train-, val- & test-set...')
        self.split_set(df)

        if Config.debug_mode:
            section_text('Tokenize with own SentPiece vocabulary II/II...')
        self._train_spm_model(df)
        self.tokenize_own_vocab(df=df, cpu_cores=cpu_cores)

        if Config.debug_mode:
            section_text('Compute meta data of tokenization...')
        self.calc_token_meta_data(df)

        if Config.debug_mode:
            section_text('Filter submissions with too much tokens or sentences...')
        self.filter_submissions(df, token_limit)

        return df
