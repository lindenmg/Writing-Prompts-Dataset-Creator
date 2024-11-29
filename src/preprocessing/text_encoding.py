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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook as tqdm
from transformers import AutoModel

from src.utils.data_processing import DataProcessor
from src.utils.hdfs_caching import HdfsCacher
from src.utils.helpers import set_seeds, round_to_next_multiple
from src.utils.settings import Config
from src.utils.sqlite import SqliteKeyValStore


class SentenceEncoder:

    # ToDo: Make limits independent from user/function input
    def __init__(self, batch_size: int
                 , story_token_limit: int
                 , story_max_num_sentences: int
                 , story_max_sentence_length: int
                 , path_kv_store: str
                 , text_src_meta='bpe_522w_768t'
                 , l_features=['mean_embeddings']):
        if story_token_limit < 1 or story_token_limit % 256 != 0:
            raise ValueError(f"`story_token_limit` should be a positive "
                             f"multiple of 256 due to DNN input "
                             f"block-size, but is {story_token_limit}!")
        if 'CLS' not in l_features and 'max_tokens' not in l_features \
                and 'mean_embeddings' not in l_features \
                and 'max_embeddings' not in l_features \
                and 'mean_tokens' not in l_features:
            raise ValueError(f"No of the given feature parameters in `_features` is valid:"
                             f" {l_features}. Use ['CLS', 'max_tokens', 'max_embeddings'"
                             f", mean_embeddings, mean_tokens]!")

        # Round up to next multiple of 8 because of TensorCores
        self.max_input_sentence_len = int(round_to_next_multiple(story_max_sentence_length, 8))
        self.is_model_loaded = False
        self.encoder_model = None
        self.batch_size = batch_size

        device = "cuda:0" if (torch.cuda.is_available() and Config.hardware.cuda) else "cpu"
        self.device = torch.device(device)

        # Next power of two greater than the longest PROMPT
        self.max_prompt_len = 128

        # BERT padding token ID for embedding
        self.PAD_ID = 0

        # BERT: For attention masking, gets added to the token embeddings
        self.MASK_P = 0
        self.MASK_T = 1

        if 'CLS' not in l_features and 'max_tokens' not in l_features \
                and 'mean_tokens' not in l_features:
            self.uses_only_embedding = True
        else:
            self.uses_only_embedding = False
        self._encoder_intern = None

        set_seeds(seed=1337)
        self.cpu_cores = Config.hardware.n_cpu
        self.l_features = l_features

        self.d_prompt_feat_config = {
            'DNN': 'BERT-Large',
            'Feats': l_features,
            'Feat-src': 'Prompt',
            'Token-limit': str(story_token_limit),
            'Text-src-meta': text_src_meta
        }

        self.d_story_feat_config = {
            'DNN': 'BERT-Large',
            'Feats': l_features,
            'Feat-src': 'Story',
            'Token-limit': str(story_token_limit),
            'Sentence-limit': str(story_max_num_sentences),
            'Text-src-meta': text_src_meta
        }
        dp = DataProcessor()
        d_hashing_for_version_control = {**self.d_story_feat_config, **self.d_prompt_feat_config}
        self.story_table_name = f"story_feats__{dp.hash_dict(d_hashing_for_version_control)}"
        self.story_max_num_sentences = story_max_num_sentences
        self.path_kv_store = path_kv_store

    def _create_bert_input(self, tokens, max_input_len=128):
        l_tokens = list(map(int, tokens.split()))
        len_ = len(l_tokens)
        assert len_ <= max_input_len

        # We pad our token ids until they reach the max prompt length
        l_tokens += [self.PAD_ID] * (max_input_len - len_)

        # We mask real tokens with a 1 and the padding with 0
        l_mask = [self.MASK_T] * len_
        l_mask += [self.MASK_P] * (max_input_len - len_)

        feature = {
            'token_ids': torch.LongTensor([l_tokens]),
            'mask_ids': torch.LongTensor([l_mask]),
            # 'token_len': torch.LongTensor([len_])
            # /\ Actual prompt length for batch-wise padding or sorting
        }
        return feature

    def _create_bert_story_input(self, story):
        l_sentences = story.split('|▁|')
        l_sentence_feats = []
        for sentence in l_sentences:
            d_feats = self._create_bert_input(sentence, self.max_input_sentence_len)
            l_sentence_feats.append(d_feats)
        return l_sentence_feats

    def _create_bert_prompt_input(self, tokens):
        return self._create_bert_input(tokens, self.max_prompt_len)

    def _create_prompt_dataloader(self, df):
        prompt_idx = df.prompt_idx.drop_duplicates()
        prompts = df.prompt_bert_tokens.loc[prompt_idx.index]
        prompt_bert_features = prompts.apply(self._create_bert_prompt_input)

        t_input_ids = torch.cat([f['token_ids'] for f in prompt_bert_features])
        t_mask_ids = torch.cat([f['mask_ids'] for f in prompt_bert_features])
        prompt_idx = prompt_idx.values.astype(dtype=np.float32)
        t_prompt_idx = torch.from_numpy(prompt_idx).reshape((-1, 1))
        t_prompt_idx = t_prompt_idx.to(self.device)
        t_input_ids = t_input_ids.to(self.device)
        t_mask_ids = t_mask_ids.to(self.device)

        dataset = TensorDataset(t_prompt_idx, t_input_ids, t_mask_ids)
        data_loader = DataLoader(dataset=dataset
                                 , batch_size=self.batch_size
                                 , shuffle=False
                                 , pin_memory=False
                                 , drop_last=False)
        return data_loader

    def _create_story_dataloader(self, df):
        """
        We concat padded & masked sentences to one long sequence
        over all stories. Then reconstruct each prompt-story pair
        through prompt_id, story_id and sentence number in the form
        of a hierarchical index.
        So we can store everything in a 2D table later on.

        Parameters
        ----------
        df: pandas.DataFrame
            With the story BERT tokens and their prompt id
            in the respective columns 'story_bert_tokens'
            and 'prompt_idx'

        Returns
        -------
        torch.utils.data.dataloader.Dataloader
            For iterating over the sentence tokens with their
            attention masks, prompt-, story- and sentence ids.
        """
        # ToDo: Sort by story_idx beforehand so it is sorted in the database after index
        prompt_idx = df.prompt_idx.values.astype(dtype=np.float32)
        story_idx = df.story_idx.values.astype(dtype=np.float32)
        story_bert_features = df.story_bert_tokens.apply(self._create_bert_story_input)

        t_input_ids = torch.cat([
            sent_feat['token_ids']
            for l_sentence_feats in story_bert_features
            for sent_feat in l_sentence_feats
        ])
        t_mask_ids = torch.cat([
            sent_feat['mask_ids']
            for l_sentence_feats in story_bert_features
            for sent_feat in l_sentence_feats
        ])
        t_prompt_idx = torch.from_numpy(prompt_idx)
        t_story_idx = torch.from_numpy(story_idx)

        # We repeat the t_*_ids by the amount of sentences for each story.
        # We do this to create the hierarchical index later on.
        # Also, each story gets ids for its sentences
        l_sentence_count = [
            len(l_feats) for l_feats in story_bert_features
        ]
        t_sentence_idx = torch.cat([
            torch.arange(sent_num) for sent_num in l_sentence_count
        ])
        l_sentence_count = torch.LongTensor(l_sentence_count)
        t_prompt_idx = t_prompt_idx.repeat_interleave(l_sentence_count).reshape((-1, 1))
        t_story_idx = t_story_idx.repeat_interleave(l_sentence_count).reshape((-1, 1))

        dataset = TensorDataset(t_prompt_idx
                                , t_story_idx
                                , t_sentence_idx
                                , t_input_ids
                                , t_mask_ids)
        data_loader = DataLoader(dataset=dataset
                                 , batch_size=self.batch_size
                                 , shuffle=False
                                 , pin_memory=False
                                 , drop_last=False)
        return data_loader

    def __bert_embeddings(self, input_ids):
        token_type_ids = torch.zeros(input_ids.size()
                                     , dtype=torch.long
                                     , device=input_ids.device)
        embed_out = self._encoder_intern.embeddings(input_ids=input_ids
                                                    , position_ids=None
                                                    , token_type_ids=token_type_ids
                                                    , inputs_embeds=None)
        # Reconstruct shape of normal BERT forward function.
        # Add some Nones, so it crashes more certainly,
        # if it wants to read from the wrong position
        return None, None, (embed_out, None)

    def _load_bert_model(self):
        if not self.is_model_loaded:
            model_pretrained = AutoModel.from_pretrained(Config.path.pretrained_model,
                                                         cache_dir=Config.path.model_folder)
            model_pretrained.to(self.device)
            model_pretrained.eval()
            if self.uses_only_embedding:
                self._encoder_intern = model_pretrained
                self.encoder_model = self.__bert_embeddings
            else:
                self.encoder_model = model_pretrained
            self.is_model_loaded = True
        return self.encoder_model

    @staticmethod
    def __compute_mean_tokens(l_vector, features, mask_ids, layer):
        # ToDo: Doesn't take the CLS token into account, but still the SEP token - Fix that
        tokens = features[2][layer][:, 1:]
        mask = mask_ids[:, 1:]
        tokens[mask == 0] = 0
        tokens = tokens.sum(dim=1) / (mask == 1).sum(dim=1).unsqueeze(dim=1).to(tokens.dtype)
        l_vector.append(tokens)

    @staticmethod
    def __compute_max_tokens(l_vector, features, mask_ids, layer):
        tokens = features[2][layer][:, 1:]
        mask = mask_ids[:, 1:]
        tokens[mask == 0] = float('-inf')
        l_vector.append(torch.max(tokens, dim=1).values)

    def _extract_features(self, features, mask_ids):
        l_vector = []
        layer = -1  # Indeed the 2nd last layer, because the last comes separate

        if 'CLS' in self.l_features:
            # CLS embedding of 2nd last layer
            cls = features[2][layer][:, 0]
            l_vector.append(cls)

        if 'max_embeddings' in self.l_features:
            # Maximum of token embeddings feature dimension
            self.__compute_max_tokens(l_vector, features, mask_ids, layer=0)

        if 'max_tokens' in self.l_features:
            # Maximum of each feature data point of the 2nd last layer token embeddings
            self.__compute_max_tokens(l_vector, features, mask_ids, layer=layer)

        if 'mean_embeddings' in self.l_features:
            # Mean of the token embeddings feature dimension:
            self.__compute_mean_tokens(l_vector, features, mask_ids, layer=0)

        if 'mean_tokens' in self.l_features:
            # Mean of 2nd last layer token feature dimension:
            self.__compute_mean_tokens(l_vector, features, mask_ids, layer=layer)

        if 'avg_given' in self.l_features:
            for i in range(len(l_vector)):
                l_vector[i] = l_vector[i].unsqueeze(dim=1)
            features = torch.mean(torch.cat(l_vector, dim=1), dim=1)
        else:
            features = torch.cat(l_vector, dim=1)
        return features.cpu().numpy()

    def _compute_prompt_features(self, model, data_loader, cacher):
        rows_total = len(data_loader.dataset)
        if Config.debug_mode:
            print(f"Computing features for {rows_total:,d} rows of text")

        with torch.no_grad():
            for prompt_ids, input_ids, mask_ids in tqdm(data_loader):
                features = model(input_ids=input_ids, attention_mask=mask_ids)
                features = self._extract_features(features, mask_ids)
                prompt_ids = prompt_ids.view(-1).int().cpu().numpy()
                df_feats = pd.DataFrame(features, index=prompt_ids)
                df_feats.index.name = 'prompt_idx'
                cacher.append_hdfs_cache(df_feats
                                         , task='prompt_feats'
                                         , d_sub_settings=self.d_prompt_feat_config
                                         , stage='p'
                                         , rows_total=rows_total)

    def encode_prompts(self, df):
        cacher = HdfsCacher()
        data_loader = self._create_prompt_dataloader(df)
        model = self._load_bert_model()
        self._compute_prompt_features(model, data_loader, cacher)
        cacher.finalize_hdfs_caching(task='prompt_feats'
                                     , d_sub_settings=self.d_prompt_feat_config
                                     , stage='p')
        cacher.clean_hdfs()

    @staticmethod
    def _save_sentence_features(kv_store, df_buffer, idx_names, store_all=False):
        """
        Stores the completed story sentence features in a DB and return the remainder

        Parameters
        ----------
        kv_store: SqliteKeyValStore
        df_buffer: pandas.DataFrame
            The table with the sentence features as values and
            the story- & sentence ids as index.
        idx_names: list of str

        Returns
        -------
        pandas.DataFrame
            Last story with its sentence features, as it might not be complete yet.
            This is because we store story sentence features as one block
        """
        end_idx = None if store_all else -1
        l_data = [
            (group.index, group.values)
            for _, group in df_buffer.groupby(by='story_idx'
                                              , axis='index'
                                              , sort=False)
        ]
        l_story_idx = [(idx.levels[0], idx.codes[0][0]) for idx, _ in l_data[0:end_idx]]
        l_story_idx = [lvl[code] for lvl, code in l_story_idx]

        if not store_all:
            # We restore the MultiIndex of the last element from the
            # levels- and codes attributes of the former one
            index = l_data[-1][0].get_values()
            multi_index = pd.MultiIndex.from_tuples(index, names=idx_names)
            df_buffer = pd.DataFrame(l_data[-1][1], index=multi_index)
        else:
            df_buffer = None
        l_data = [value for _, value in l_data[0:end_idx]]
        l_keys = [str(k) for k in l_story_idx]
        kv_store.write_batch(l_keys=l_keys, l_data=l_data, overwrite=False)
        return df_buffer

    def _compute_story_features(self, model, data_loader, kv_store):
        """
        Compute BERT feature vectors of the individual story sentences

        For storing the features in SQLite, we use SQLiteDict to access
        the database in key-value store fashion. High-level layout:
             KEY       ||                     DATA
        '<story_idx>'  ||  <numpy.ndarray> with shape (sentences, features)

        Parameters
        ----------
        model: torch.module
            The pretrained BERT model
        data_loader: torch.utils.data.dataloader.Dataloader
            For iterating over the sentence tokens with their
            attention masks, prompt-, story- and sentence ids.
        kv_store: SqliteKeyValStore
            For storing the sentence feature vectors in key=story_idx
            val=`sentence features` form
        See Also
        ---------
        https://pypi.org/project/sqlitedict/
        """
        save_interval = math.ceil(4 * self.story_max_num_sentences / self.batch_size)
        rows_total = self.batch_size * len(data_loader)
        idx_names = ['story_idx', 'sentence_idx']

        if Config.debug_mode:
            print(f"Computing features for {rows_total:,d} rows of text")

        with torch.no_grad():
            df_buffer = None

            for i, (_, story_idx, sent_idx, input_ids, mask_ids) in enumerate(tqdm(data_loader)):
                input_ids = input_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                features = model(input_ids=input_ids, attention_mask=mask_ids)
                features = self._extract_features(features, mask_ids)
                l_idx = [story_idx.view(-1).int(), sent_idx.view(-1).int()]
                multi_index = pd.MultiIndex.from_arrays(l_idx, names=idx_names)
                df_feats = pd.DataFrame(features, index=multi_index)

                if df_buffer is not None:
                    df_buffer = df_buffer.append(df_feats, verify_integrity=True)
                    df_feats = None
                else:
                    df_buffer = df_feats

                if (i + 1) % save_interval == 0:
                    df_buffer = self._save_sentence_features(kv_store, df_buffer, idx_names)
            self._save_sentence_features(kv_store, df_buffer, idx_names, store_all=True)

    def encode_stories(self, df):
        """
        Save feature vectors of stories sentences in a 2D HDFS table

        Parameters
        ----------
        df: pandas.DataFrame
            With the story BERT tokens and their prompt id
            in the respective columns 'story_bert_tokens'
            and 'prompt_idx'

        See Also
        ---------
        SentenceEncoder._compute_story_features([..]) for an explanation
        of the hierarchical multi-index for accessing the feature vectors.
        """
        kv_store = SqliteKeyValStore(self.path_kv_store, table=self.story_table_name)
        data_loader = self._create_story_dataloader(df)
        model = self._load_bert_model()
        self._compute_story_features(model=model, data_loader=data_loader, kv_store=kv_store)
        kv_store.close()

        if Config.debug_mode:
            print("FINISHED computation of story sentence features!")
