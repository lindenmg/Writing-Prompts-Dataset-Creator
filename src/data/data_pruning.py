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
import unicodedata

from collections import Counter
from pathlib import Path
from pyfasttext import FastText

from src.data.spacyoverlay import SpacyOverlay
from src.utils.helpers import parallel_pandas, section_text
from src.utils.settings import Config


# ToDo: Look at oov words for author names and
#       remove respective text lines in the story texts !
class DataPruner:

    def __init__(self, ):
        """
        For pruning data from r/WritingPrompts subreddit.

        This class has a prune_data function which will
        concurrently process a Pandas DataFrame and remove
        deleted submissions where just a mod message remains.
        It will also remove stories flagged as NSW/NSFL.

        And it will delete substrings from the submissions
        which are author messages. And it will replace URLs
        with the tag <URL>
        """
        # === Profanity Word List ============================================

        path_swear = \
            Path(Config.path.data_folder) / \
            'external' / \
            'profanity_words.txt'

        if not path_swear.is_file():
            raise FileNotFoundError()
        with open(str(path_swear), "r") as f:
            self.swear_words = f.readlines()

        self.ptrn_tokenize = re.compile(r'(\w+|[^\w\s])', flags=re.IGNORECASE)
        self.swear_words = [
            self.tokenize(w.strip().lower())
            for w in self.swear_words
        ]

        # === Unwanted Unicode characters ====================================

        path_strange = \
            Path(Config.path.data_folder) / \
            'external' / \
            'strange_characters.txt'

        if not path_strange.is_file():
            raise FileNotFoundError()
        with open(str(path_strange), "r") as f:
            self.strange_chars = f.readlines()
        self.strange_chars = [char.strip() for char in self.strange_chars]

        # === Non-English language stories ===================================

        self.ft_predict = None
        self.path_lang = \
            Path(Config.path.data_folder) / \
            'external' / \
            'lid.176.bin'

        if not self.path_lang.is_file():
            raise FileNotFoundError('fastText language detection'
                                    ' model lid.176.bin not found! '
                                    'Please follow Readme.md instructions')

        spacy = SpacyOverlay()
        self.nlp = spacy.get_nlp()

        # === Regular expressions for deleting submissions ===================

        # Find messages from mods which state that this story has been removed
        # - or that it is off-topic (no story)
        self.ptrn_mod_del = re.compile(
            r'(?:this submission has been removed|'
            r'post has been removed|^Removed|'
            r'This post was removed|I have removed your post|'
            r'(your|this) prompt was removed|'
            r'prompt has been removed|How To Tag Prompts|'
            r'top-level comments must be a story or poem\. Reply|'
            r'moderators reserve the right to|'
            r'\*\*Off-Topic Discussion\*\*|^.*\[OT\])'
            , flags=re.IGNORECASE)

        # Find stories which got flagged as NSFW/NSFL by the author
        # NSFW == Not safe for work; NSFL == Not safe for life
        # Those tags indicate mostly sexual content or extreme violence
        self.ptrn_nsfw = re.compile(r'^(?:.*)\b(?:NSFW|NSWF|NSFL)\b(?:.*\n)'
                                    , flags=re.IGNORECASE)

        # ====================================================================
        # === Regular expressions for text substitution ===

        # --- URL ------------------------------------------------------------

        # Find URLs in the Reddit text format [hyperlink-text](URL)
        # Source: https://stackoverflow.com/a/28552670
        # Source consulted at June 15, 2019
        self.ptrn_format_url = re.compile(
            r'\(\s?(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+).*?\)'
            , flags=re.IGNORECASE)

        # Find not formatted URLs
        # Source: https://stackoverflow.com/a/30408189
        # Source consulted at June 15, 2019
        self.ptrn_url_tag = re.compile(
            r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?'
            r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'
            r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}'
            r'(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))'
            r'|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)'
            r'(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*'
            r'(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s\]\)]*)?'
            , flags=re.IGNORECASE)

        # --- AUTHOR MESSAGES ------------------------------------------------

        # Find author messages at the beginning of a story
        self.ptrn_msg_b = re.compile(r"^.{,10}(author'?s? note|A/N|"
                                     r"note:|Disclaimer|I wrote this).*\n"
                                     , flags=re.IGNORECASE)

        # Find author messages at the end of a story
        self.ptrn_msg_e = re.compile(r"(?:\n.*(author'?s? note|A/N|note:|"
                                     r"/?\br/\w+|Thank.* for reading|"
                                     r"new to (writing|this sub)|"
                                     r"Much appreciated|\bedit.?\:|Disclaimer|"
                                     r"feedback is appreciated|on mobile|"
                                     r"\bEdits|criticism (welcome|appreciated)|"
                                     r"at my subreddit|Might continue later|"
                                     r"If you have enjoyed this|Check out.*\[|"
                                     r"subscribing to.*(subreddit|\[)).*)$"
                                     , flags=re.IGNORECASE)

        # Find declaration of 'being new to this' at the beginning of a story
        self.ptrn_newto_b = re.compile(r"^.*(new to (writing|this sub)).*\n"
                                       , flags=re.IGNORECASE)

        # Find substrings which indicate further author messages
        # at the beginning of a story
        self.ptrn_com_b = re.compile(r'^(?:.*(feedback is appreciated'
                                     r'|on mobile|criticism (welcome|'
                                     r'appreciated)).*\n)'
                                     , flags=re.IGNORECASE)

        # Find an author message at the beginning of a story OR prompt_body
        self.ptrn_uni_msg_b = re.compile(r'^(?:.{,90})(\bedit.?\:|'
                                         r'First writing|first WP)(?:.*(\n|$))'
                                         , flags=re.IGNORECASE)

        # Find lines which are 'Edit' messages
        self.ptrn_edit_l = re.compile(r'(?:\n.{,3}\bedit\d?\b.*)'
                                      , flags=re.IGNORECASE)

        # Find a note which gives the credit to someone else in a prompt_body
        self.ptrn_credit = re.compile(r'Credit to.*(\n|$)'
                                      , flags=re.IGNORECASE)

        # ====================================================================
        # === Regular expressions for final whitespace cleaning ===

        # Find line-breaks, which have intermediate whitespaces
        # or are cover more than one blank line
        self.ptrn_linebreak = re.compile(r'\s*\n\s*\n+\s*')

        # Find multiple whitespaces, except linefeed (line-breaks)
        self.ptrn_ws = re.compile(r'[^\S\n]+')

        # Find special characters lines at the beginning & end of a text
        self.ptrn_char_void = re.compile(r'^[_\W]+\n|\n[_\W]+$')

    def tokenize(self, text):
        token_text = ' '.join(re.findall(self.ptrn_tokenize, text.lower()))
        return ' ' + token_text + ' '

    def search_profanities(self, text):
        token_text = self.tokenize(text)

        for word in self.swear_words:
            if word in token_text:
                return True
        return False

    def flag_for_removal(self, text: str):
        """
        Returns
        -------
        bool
            True, if the text contains a moderator message which states that
            this submission/post has been removed. Or if it is an off-topic
            discussion. Or if the submission is flagged as NSFW/NSFL.
            Or if it contains at least one profanity word. Or if it has
            strange characters (symbols/diacritics) in it.
            False, else.
        """
        has_strange_chars = False
        profanities = False

        # Check, if this submission has apparently been deleted
        deleted = re.search(self.ptrn_mod_del, text)
        nsfw = None

        if deleted is None:
            profanities = self.search_profanities(text)

        if deleted is None and not profanities:
            for sign in self.strange_chars:
                if sign in text:
                    has_strange_chars = True
                    break

        if deleted is None and not profanities and not has_strange_chars:
            nsfw = re.search(self.ptrn_nsfw, text)

        return deleted is not None \
            or nsfw is not None \
            or profanities \
            or has_strange_chars

    def prune_prompt_body(self, text: str):
        """
        Returns
        -------
        str
            String whose content got processed in the following manner:
            Removed 'Credit to [..]' messages and the '[removed]' flag
            which indicates a removed prompt body
        """
        text = text.replace('[removed]', '')
        text = text.replace('[deleted]', '')
        return re.sub(self.ptrn_credit, '', text)

    def prune_submission(self, text: str):
        """
        Returns
        -------
        str
            String whose content got processed in the following manner:
            URL in the last line removed if there was one.
            remaining URLs swapped with string tag '<URL>'
            First lines which are edit messages of the author removed
            Lines which are edit messages of the author removed.

            There is a low possibility of false positives regarding
            the removal. Obviously false negatives can also occur.
        """
        text = re.sub(self.ptrn_format_url, '(<URL>)', text)
        text = re.sub(self.ptrn_url_tag, '<URL>', text)
        text = re.sub(self.ptrn_edit_l, '', text)
        return re.sub(self.ptrn_uni_msg_b, '', text)

    def prune_story_text(self, text: str):
        """
        Returns
        -------
        str
            Remove the first and last line if they contain messages
            of the author addressed to the reader (no story-content).
            There is a low possibility of false positives regarding
            the removal. Obviously false negatives can also occur.
        """
        text = re.sub(self.ptrn_msg_b, '', text)
        text = re.sub(self.ptrn_com_b, '', text)
        text = re.sub(self.ptrn_msg_e, '', text)
        text = text.replace('### For More Legends From The Multiverse', '')
        return re.sub(self.ptrn_newto_b, '', text)

    def detect_lang_ext(self, text):
        doc = self.nlp(text)
        l_lang = []
        try:
            for sent in doc.sents:
                l_lang.append(self.ft_predict(str(sent), k=1)[0][0])
        except ValueError:
            # In case of sentence detection failure
            # we don't want a division by zero. We
            # also want to exclude this sample totally
            l_lang = ['Error']
        counter = Counter(l_lang)

        # Simple heuristic which works half-way:
        # Check if apparently more than 1/3 of the sentences are not English
        return (counter.get('en', 0) / len(l_lang)) < 2 / 3

    @staticmethod
    def _data_removal(df_use, df, removal_func, column: str, cpu_cores: int):
        if cpu_cores > 1:
            parallel_pandas(func=removal_func, df=df_use, column=column
                            , new_column='remove', num_threads=cpu_cores)
        elif cpu_cores == 1:
            df_use['remove'] = df_use[column].apply(lambda s: removal_func(s))
        else:
            raise ValueError(f"Number of CPU cores must be "
                             f">= 1, but is {cpu_cores}")

        idx = df_use[df_use['remove']].index
        df.drop(idx, axis='index', inplace=True)
        df_use.drop(labels='remove', axis='columns', inplace=True)
        del idx
        gc.collect()

    def _prune_stories(self, df, cpu_cores):
        parallel_pandas(func=self.prune_submission
                        , df=df
                        , column="story"
                        , num_threads=cpu_cores)
        parallel_pandas(func=self.prune_story_text
                        , df=df
                        , column="story"
                        , num_threads=cpu_cores)

        # Delete all submissions with now empty stories
        idx = df[df['story'] == ''].index
        df.drop(idx, axis='index', inplace=True)
        del idx
        gc.collect()

        # Sort out a good part of the non-English / nonsense text stories:
        # We have to work around the fact that the FatText class brakes the
        # multiprocessing Pool functionality (parallel_pandas function)
        ft_model = FastText(str(self.path_lang))
        self.ft_predict = ft_model.predict_proba_single
        df['story_lang'] = df['story'].apply(
            lambda s: self.ft_predict(s, k=1)[0][0]
        )
        idx = df[df['story_lang'] != 'en'].index
        df_lang = df.loc[idx, :]
        self._data_removal(df_lang, df, self.detect_lang_ext, 'story', 1)
        df.drop(labels='story_lang', axis='columns', inplace=True)
        del self.ft_predict
        del df_lang

        # Drop duplicated stories
        idx = df[df.duplicated(subset='story')].index
        df.drop(idx, axis='index', inplace=True)
        gc.collect()

    def clean_text(self, text: str):
        # Reduce multiple empty lines with possible
        # whitespaces in between to one emtpy line
        text = re.sub(self.ptrn_linebreak, '\n\n', text)

        # Remove special character lines at the start & end of a text
        text = re.sub(self.ptrn_char_void, '', text)

        # Reduce multiple whitespaces except linefeed to one space
        text = re.sub(self.ptrn_ws, ' ', text)

        # Here we strip all accents from the text (Mostly because of XLNet
        # whereas the previous version just replaced most of them)
        # We have to do it after the pruning, so we can still remove texts
        # with those diacritics which indicate foreign languages or strange words
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([char for char in text if not unicodedata.combining(char)])
        return text.strip()

    def prune_data(self, df, cpu_cores=2):
        """
        Delete removed submissions, prune author notes, etc.

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame which contains the WritingPrompt text corpus
            Should contain 'prompt', 'prompt_body' & 'story' string columns
            Call by reference return!
        cpu_cores: int
            Number of physical CPU cores to use. NO hyper-threading!
        """
        if 'story' not in df.columns \
                or 'prompt_body' not in df.columns \
                or 'prompt' not in df.columns:
            raise ValueError(f"``df`` should contain at least string columns "
                             f"'story', 'prompt' & 'prompt_body, but does "
                             f"contain {df.columns}")

        if Config.debug_mode:
            section_text("Removing deleted & NSFW & "
                         "profanity words submissions...")
        self._data_removal(df, df, self.flag_for_removal, 'story', cpu_cores)
        self._data_removal(df, df, self.search_profanities, 'prompt', cpu_cores)

        if Config.debug_mode:
            section_text("Pruning story text...")
        self._prune_stories(df, cpu_cores)

        if Config.debug_mode:
            section_text("Pruning prompt body text...")
        parallel_pandas(func=self.prune_submission
                        , df=df
                        , column="prompt_body"
                        , num_threads=cpu_cores)
        parallel_pandas(func=self.prune_prompt_body
                        , df=df
                        , column="prompt_body"
                        , num_threads=cpu_cores)

        if Config.debug_mode:
            section_text("Pruning prompt title text...")
        df['prompt'] = df['prompt'].apply(
            lambda s: re.sub(self.ptrn_url_tag, '<URL>', s)
        )

        if Config.debug_mode:
            section_text("Final clean up of whitespaces & accents...")
        parallel_pandas(func=self.clean_text, df=df
                        , column="story", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_text, df=df
                        , column="prompt_body", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_text, df=df
                        , column="prompt", num_threads=cpu_cores)
        gc.collect()
