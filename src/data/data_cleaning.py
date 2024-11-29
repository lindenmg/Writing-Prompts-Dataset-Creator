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

from ftfy import fix_text

from src.utils.helpers import parallel_pandas, section_text
from src.utils.settings import Config


class DataCleaner:

    def __init__(self, ):
        """
        For cleaning data from r/WritingPrompts subreddit.

        This class has a clean_text function which will
        concurrently process a Pandas DataFrame and remove
        among other things multiple whitespaces, paragraph
        separators made out of special characters. It will
        normalize special characters which indicate speech.
        And reduce special characters to three occurrences
        in a row...
        """

        # ==== Regular expressions for data cleaning. Used for substitution ====
        # --- PROMPTS ONLY ---
        # Find all prompt tags, lower- or uppercase
        self.ptrn_tag = re.compile(r'\[(WP|SP|RF|TT|EU|CC|OT|IP|MP|PI|PM|CW)\]'
                                   , flags=re.IGNORECASE)

        # Find all (wanted) prompt tags which are not at the prompt beginning
        self.ptrn_tag_wp = re.compile(r'(.+?)(\[WP\])')
        self.ptrn_tag_sp = re.compile(r'(.+?)(\[SP\])')
        self.ptrn_tag_rf = re.compile(r'(.+?)(\[RF\])')
        self.ptrn_tag_tt = re.compile(r'(.+?)(\[TT\])')

        # Find all wanted prompt tags which have a following colon
        self.ptrn_tag_fc = re.compile(r'(\[(WP|SP|RF|TT)\]):')

        # Find all multiple prompt tags of same type
        self.ptrn_tag_multi = re.compile(r'(\[(WP|SP|RF|TT)\])((.*)\1+)+')

        # Find all wanted prompt tags at the beginning which may have a following colon
        self.ptrn_tag_ws = re.compile(r'((\[(WP|SP|RF|TT)\])+)([^\[\s])')

        # --- WHITESPACE CLEANING ---
        # Find multiple whitespaces, except linefeed (line-breaks)
        self.ptrn_ws = re.compile(r'[^\S\n]+|&nbsp;')

        # Find zero-width whitespaces or equivalent
        self.ptrn_zero_ws = re.compile(r'<Enter>|&#x200B;|\u200b|\uFEFF|\u200c|\u200d'
                                       , flags=re.IGNORECASE)

        # Find broken ellipsis - which has only .. instead of ...
        self.ptrn_ellips = re.compile(r'([^\.]|^)\.\.([^\.]|$)')

        # Find line-breaks, which have intermediate whitespaces
        # or are cover more than one blank line
        self.ptrn_linebreak = re.compile(r'\s*\n\s*\n+\s*')

        # Find whitespaces before or after a linefeed (line-breaks)
        self.ptrn_lb_ws = re.compile(r'\n ')
        self.ptrn_ws_lb = re.compile(r' \n')

        # --- SPECIAL CHARACTERS ---
        # Find all Unicode dashes but all that could visually be used instead of '-'
        # Which doesn't mean that they are looking the same, they have just the same function
        self.ptrn_dash = re.compile(r'[\－\-\–\—\﹣\⸗\―\–\‒\‑\‐\־\֊\᠆\⸺\⸻\-\─\−\⁃]+')

        # Find different kind of apostrophes
        self.ptrn_apos = re.compile(r"’|‘|´|`|᾽|‛|′|،|ˈ|ʻ|ˈ")

        # Find different kind of quotes
        self.ptrn_quote = re.compile(r'“|”|“|„|¨|″|˝')

        # Find special characters which are used to separate paragraphs
        self.ptrn_sep_char = re.compile(r'\n[_\W]+\n')

        # Find special characters lines at the beginning & end of a text
        self.ptrn_char_void = re.compile(r'^[_\W]+\n|\n[_\W]+$')

        # Find special characters of the same kind which occur more than 3 times in a row
        self.ptrn_char_row = re.compile(r'([^\s\w])\1{3,}')

        # Find alphanumeric characters which have a '^' in front
        self.ptrn_caret_word = re.compile(r'\^(\w)')

        # --- OTHER ---
        # Find honorifics which are not separated with a whitespace from the following name
        self.ptrn_title_ws = re.compile(r'(Ms|Mr|Mrs|Dr|Prof)\.(\w)')

        # Misspellings & Unicode characters with similar looks
        self.ptrn_fiance = re.compile(r'fiancé(\W)', flags=re.IGNORECASE)

        # Find '^' which are used as 'power of' math symbol
        self.ptrn_power = re.compile(r'(\w)\^(\d)')

        # Same letters that occur more than three times in a row
        self.ptrn_letter_row = re.compile(r'([^0-9\W])\1{3,}')

    def clean_prompt(self, prompt: str):
        # Convert all WritingPrompt category tags to uppercase
        prompt = re.sub(self.ptrn_tag
                        , lambda m: '[' + m.group(1).upper() + ']'
                        , prompt)

        # Erase all colons after a (wanted) WP category tag
        prompt = re.sub(self.ptrn_tag_fc, r'\1', prompt)

        # Simple prompt candidate according to r/WritingPrompts rules as of 06/2019
        # We set a [SP] tag to the prompts with less than 100 characters total.
        # Of course, we will not get it completely correct, but that is okay.
        if len(prompt) < 101:
            if '[SP]' in prompt and '[WP]' in prompt:
                prompt = prompt.replace('[WP]', '')
            elif '[SP]' not in prompt and '[WP]' in prompt:
                prompt = prompt.replace('[WP]', '[SP]')
            elif '[SP]' not in prompt and len(prompt) < 97:
                prompt = '[SP]' + prompt

        # Move all (wanted) WP category tags to the front of the prompt
        prompt = re.sub(self.ptrn_tag_tt, r'\2\1', prompt)
        prompt = re.sub(self.ptrn_tag_rf, r'\2\1', prompt)
        prompt = re.sub(self.ptrn_tag_sp, r'\2\1', prompt)
        prompt = re.sub(self.ptrn_tag_wp, r'\2\1', prompt)

        # Reduce multiplied tags to one of their kind. E.g. [WP][WP] --> [WP]
        prompt = re.sub(self.ptrn_tag_multi, r'\1\4', prompt)
        prompt = re.sub(self.ptrn_tag_multi, r'\1\4', prompt)

        # Insert a (further) whitespace between WP category tag and text
        return re.sub(self.ptrn_tag_ws, r'\1 \4', prompt)

    def clean_spaces(self, text: str):
        # Remove zero-width whitespaces & equivalents
        text = re.sub(self.ptrn_zero_ws, '', text)

        # Reduce multiple whitespaces except linefeed to one space
        text = re.sub(self.ptrn_ws, ' ', text)
        return text.strip()

    def clean_linebreaks(self, text: str):
        # Reduce multiple empty lines with possible whitespaces
        # in between to one emtpy line
        text = re.sub(self.ptrn_linebreak, '\n\n', text)

        # Remove whitespaces before & after a linefeed (linebreak)
        text = re.sub(self.ptrn_lb_ws, '\n', text)
        text = re.sub(self.ptrn_ws_lb, '\n', text)

        # Remove special characters which are used as paragraph separators
        text = re.sub(self.ptrn_sep_char, '\n\n', text)

        # Remove special character lines at the start & end of a text
        return re.sub(self.ptrn_char_void, '', text)

    def clean_text(self, text: str):
        # Has to be done before dash-replacement
        text = text.replace('----ing', 'fucking')

        # Reduce (multiple) Unicode dashes to standard dash
        text = re.sub(self.ptrn_dash, '-', text)

        # Normalizes apostrophes and similar signs
        text = re.sub(self.ptrn_apos, "'", text)

        # Normalize quotes and similar signs
        text = re.sub(self.ptrn_quote, '"', text)

        # Replace some special chars with plainer ones
        text = text.replace('。', ".").replace("''", '"').replace('…', "...")
        text = text.replace('»', '"').replace('«', '"')

        # Replace '^' as math 'power of' sign with '**' like in Python
        text = re.sub(self.ptrn_power, r'\1**\2', text)

        # Remove '^' before alphanumeric characters
        text = re.sub(self.ptrn_caret_word, r'\1', text)

        # Fix a few misspellings (yes, that's all)
        text = re.sub(self.ptrn_fiance, r'fiancee\1', text)
        text = text.replace('Caffè', 'Cafe')
        text = text.replace('supervillan', 'supervillain')
        text = text.replace('tablette', 'tablet')
        text = text.replace('tablett', 'tablet')
        text = text.replace('timeloop', 'time loop')
        text = text.replace('no-one', 'no one')
        text = text.replace('No-one', 'No one')
        text = text.replace('emenating', 'emanating')
        text = text.replace('writingprompts', 'WritingPrompts')

        # Fix a few non-regular letters/characters
        text = text.replace('½', '1/2').replace('≈', '=').replace('·', '*')
        text = text.replace('±', '+-').replace('÷', '/').replace('×', '*')
        text = text.replace('█', '#').replace('�', '?')

        # Replace letters which look similar like the ASCII ones.
        text = text.replace('в', 'B').replace('А', 'A').replace('Р', 'P')
        text = text.replace('ᴀ', 'A').replace('ʙ', 'B').replace('‽', '?')
        text = text.replace('ᴄ', 'C').replace('ᴅ', 'D').replace('ᴇ', 'E')
        text = text.replace('ꜰ', 'F').replace('ɢ', 'G').replace('ʜ', 'H')
        text = text.replace('ɪ', 'I').replace('ᴊ', 'J').replace('ᴋ', 'K')
        text = text.replace('ʟ', 'L').replace('ᴍ', 'M').replace('ɴ', 'N')
        text = text.replace('ǫ', 'Q').replace('ᴏ', 'O').replace('ᴘ', 'P')
        text = text.replace('ʀ', 'R').replace('ꜱ', 'S').replace('ᴛ', 'T')
        text = text.replace('ᴜ', 'U').replace('ᴠ', 'V').replace('ᴡ', 'W')
        text = text.replace('ʏ', 'Y').replace('ᴢ', 'Z').replace('ғ', 'F')
        text = text.replace('Т', 'T').replace('М', 'M')

        # Remove an author signature which occurs relatively often
        text = text.replace('######[](#dropcap)', '')

        # Replace broken ellipsis .. with ...
        text = re.sub(self.ptrn_ellips, r'\1...\2', text)

        # Reduce special characters which occur more than 3x in a row to 3
        text = re.sub(self.ptrn_char_row, r'\1\1\1', text)

        # Insert missing whitespace between honorific & name
        text = re.sub(self.ptrn_title_ws, r'\1. \2', text)

        # Reduce letters which occur more than 3x in a row to 3
        text = re.sub(self.ptrn_letter_row, r'\1\1\1', text)
        return text.strip()

    def clean_data(self, df, cpu_cores=2):
        """
        Normalizes whitespaces, linebreaks, special characters & some formatting.

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame which contains the WritingPrompt text corpus
            Should contain 'prompt', 'prompt_body' & 'story' string columns
        cpu_cores: int
            Number of physical CPU cores to use.
        """
        if 'story' not in df.columns \
                or 'prompt_body' not in df.columns \
                or 'prompt' not in df.columns:
            raise ValueError(f"``df`` should contain at least string columns "
                             f"'story', 'prompt' & 'prompt_body, but does "
                             f"contain {df.columns}")

        if Config.debug_mode:
            section_text("cleaning story text...")
        parallel_pandas(func=fix_text, df=df, column="story", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_spaces, df=df, column="story", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_linebreaks, df=df, column="story", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_text, df=df, column="story", num_threads=cpu_cores)

        if Config.debug_mode:
            section_text("cleaning prompt body text...")
        parallel_pandas(func=fix_text, df=df, column="prompt_body", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_spaces, df=df, column="prompt_body", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_linebreaks, df=df, column="prompt_body", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_text, df=df, column="prompt_body", num_threads=cpu_cores)

        if Config.debug_mode:
            section_text("cleaning prompt title text...")
        parallel_pandas(func=fix_text, df=df, column="prompt", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_spaces, df=df, column="prompt", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_prompt, df=df, column="prompt", num_threads=cpu_cores)

        # Because of newly created spaces through prompt tag moving. We need
        # it also beforehand to get rid of the zero-length whitespaces.
        parallel_pandas(func=self.clean_spaces, df=df, column="prompt", num_threads=cpu_cores)
        parallel_pandas(func=self.clean_text, df=df, column="prompt", num_threads=cpu_cores)
