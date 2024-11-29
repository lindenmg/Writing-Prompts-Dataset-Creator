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

import math
from subprocess import call

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing.text_encoding import SentenceEncoder
from src.utils.helpers import round_to_next_multiple
from src.utils.sqlite import SqliteKeyValStore

np.random.seed(1337)


class TestSentenceEncoder:

    @staticmethod
    def _gen_sentence_lengths(num_stories):
        sents_len = np.random.normal(loc=21, scale=9, size=num_stories).astype(dtype=int)
        return sents_len.clip(min=3, max=61)

    @staticmethod
    def _gen_story_idx(num_stories, sents_len, rnd_start=True):
        if rnd_start:
            start_idx = np.random.randint(low=0, high=114_000)
        else:
            start_idx = 0
        story_idx = np.arange(start=start_idx, stop=start_idx + num_stories)
        return story_idx.repeat(sents_len)

    def _create_pseudo_features(self, num_stories, num_feats=1024):
        sents_len = self._gen_sentence_lengths(num_stories)
        shape = (sents_len.sum(), num_feats)
        feats = np.random.normal(loc=0, scale=0.1, size=shape).astype(dtype=np.float32)
        story_idx = self._gen_story_idx(num_stories, sents_len)
        sentence_idx = [np.arange(len_) for len_ in sents_len]
        sentence_idx = np.hstack(sentence_idx)
        assert len(sentence_idx) == len(story_idx)
        assert len(feats) == len(sentence_idx)
        return story_idx, sentence_idx, sents_len, feats

    def _create_dataloader(self, num_prompts, num_stories, batch_size):
        sents_len = self._gen_sentence_lengths(num_stories)
        t_story_idx = self._gen_story_idx(num_stories, sents_len, rnd_start=True)
        t_sentence_idx = [np.arange(len_) for len_ in sents_len]
        t_sentence_idx = np.hstack(t_sentence_idx)
        t_sentence_idx = torch.from_numpy(t_sentence_idx).reshape((-1, 1))
        t_prompt_idx = torch.randint(low=0, high=num_prompts, size=(num_stories, 1))
        t_prompt_idx = t_prompt_idx.repeat_interleave(torch.from_numpy(sents_len))
        t_input_ids = torch.randint(low=0, high=28996, size=(len(t_story_idx), 8))
        t_mask_ids = torch.randint(low=0, high=2, size=(len(t_story_idx), 8))
        t_story_idx = torch.from_numpy(t_story_idx).reshape((-1, 1))

        dataset = TensorDataset(t_prompt_idx
                                , t_story_idx
                                , t_sentence_idx
                                , t_input_ids
                                , t_mask_ids)
        data_loader = DataLoader(dataset=dataset
                                 , batch_size=batch_size
                                 , shuffle=False
                                 , pin_memory=False
                                 , drop_last=False)
        return data_loader, t_story_idx.unique(sorted=False).cpu().numpy(), sents_len

    def test_create_bert_input(self):
        raise NotImplementedError

    def test_create_story_dataloader(self):
        path_db = '../../test_data/feats_test.db'

        df = pd.read_csv("../../test_data/test_tokenized_small.csv")
        df.reset_index(drop=True, inplace=True)

        encoder = SentenceEncoder(batch_size=64
                                  , story_token_limit=512
                                  , story_max_num_sentences=64
                                  , story_max_sentence_length=96
                                  , path_kv_store=path_db)

        dataloader = encoder._create_story_dataloader(df)
        dataset = dataloader.dataset

        assert len(dataset) == df.story_sent_num.sum()

        for t_pid, t_stid, t_seid, t_x, t_mask in dataloader:
            t_pid = t_pid.view(-1).cpu().numpy().tolist()
            t_stid = t_stid.view(-1).cpu().numpy().tolist()
            t_seid = t_seid.view(-1).cpu().numpy().tolist()
            t_x = t_x.cpu().numpy()
            t_mask = t_mask.to(dtype=torch.bool).cpu().numpy()
            for pid, stid, seid, x, mask in zip(t_pid, t_stid, t_seid, t_x, t_mask):
                p_df = df[df.prompt_idx == pid]
                s_df = p_df[p_df.story_idx == stid]
                assert len(s_df) > 0

                l_tokens = s_df.story_bert_tokens.item().split('|▁|')
                x = x[mask]
                x = " ".join(map(str, x))
                assert stid in p_df.story_idx.values
                assert (seid + 1) <= int(s_df.story_sent_num)
                assert l_tokens[seid].strip() == x
        call(['rm', path_db])

    def test_compute_story_features(self):
        batch_size = 64
        num_prompts = 543
        num_stories = 1337

        # SETUP ---------------------------------------------------------------
        path_db = '../../test_data/feats_test.db'
        encoder = SentenceEncoder(batch_size=batch_size
                                  , story_token_limit=512
                                  , story_max_num_sentences=64
                                  , story_max_sentence_length=96
                                  , path_kv_store=path_db
                                  , l_features=['mean_embeddings'])

        table = 'prompt_feats__p__932jfe832iufew9832ijf32983r298'
        dataloader, story_idx, sents_len = self._create_dataloader(num_prompts,
                                                                   num_stories,
                                                                   batch_size)
        cacher = SqliteKeyValStore(path_slite_db=path_db, table=table)
        model = FakeBert(dim_model=1024, device=encoder.device)

        # DATA CALCULATION & SAVING -------------------------------------------
        encoder._compute_story_features(model=model
                                        , data_loader=dataloader
                                        , kv_store=cacher)

        # TEST ----------------------------------------------------------------

        cacher.close()
        story_idx = np.unique(story_idx).tolist()
        store = SqliteKeyValStore(path_slite_db=path_db, table=table)
        l_data = store.read_batch(story_idx)

        # DATA TEST - actual test ---------------------------------------------
        # start = 0
        for id, sents, data in zip(story_idx, sents_len.tolist(), l_data):
            assert sents == data.shape[0]
            assert 1024 == data.shape[1]
            # end = start + sents
            # slice_ = feats[start:end]
            # start = end
            # assert (slice_ == data).all()
        call(['rm', path_db])

    def test_save_sentence_features(self, ):
        l_num = np.random.randint(low=4, high=1500, size=10)
        l_batch = np.random.choice([4, 8, 16, 32, 64], size=10)

        for test_run, (batch_size, num_stories) in enumerate(zip(l_batch, l_num)):
            # DATA PREPARATION ----------------------------------------------------
            idx_names = ['story_idx', 'sentence_idx']
            max_sents = 64
            path_db = '../../test_data/feats_test.db'
            encoder = SentenceEncoder(batch_size=batch_size
                                      , story_token_limit=512
                                      , story_max_num_sentences=max_sents
                                      , story_max_sentence_length=96
                                      , path_kv_store=path_db)
            f_save_feats = encoder._save_sentence_features
            table = encoder.story_table_name
            store = SqliteKeyValStore(path_slite_db=path_db, table=table)
            story_idx, sents_idx, sents_len, feats = self._create_pseudo_features(num_stories=num_stories)
            save_interval = math.ceil(max_sents / batch_size)
            iter_end = round_to_next_multiple(len(sents_idx), batch_size)
            df_buffer = None
            start = 0

            # DATA SAVING ---------------------------------------------------------
            for i in range(0, iter_end, batch_size):
                end = start + batch_size
                features = feats[start:end]
                l_idx = [story_idx[start:end], sents_idx[start:end]]
                multi_index = pd.MultiIndex.from_arrays(l_idx, names=idx_names)
                df_feats = pd.DataFrame(features, index=multi_index)
                start += batch_size

                if df_buffer is not None:
                    df_buffer = df_buffer.append(df_feats, verify_integrity=True)
                    df_feats = None
                else:
                    df_buffer = df_feats
                if (i + 1) % save_interval == 0:
                    df_buffer = f_save_feats(store, df_buffer, idx_names)
            f_save_feats(store, df_buffer, idx_names, store_all=True)

            # DATA READING -------------------------------------------------------
            store.close()
            story_idx = np.unique(story_idx).tolist()
            store = SqliteKeyValStore(path_slite_db=path_db, table=table)
            l_data = store.read_batch(story_idx)

            # DATA TEST - actual test ---------------------------------------------
            start = 0
            for id, sents, data in zip(story_idx, sents_len.tolist(), l_data):
                assert sents == data.shape[0]
                assert 1024 == data.shape[1]
                end = start + sents
                slice_ = feats[start:end]
                start = end
                assert (slice_ == data).all()
            print(f"passed iteration {test_run}")
            call(['rm', path_db])
