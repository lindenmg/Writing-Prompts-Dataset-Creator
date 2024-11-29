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

import spacy


class SpacyOverlay:

    def __init__(self
                 , model_type='en'
                 , pipeline_to_remove=list(['tagger', 'parser', 'ner', 'textcat'])):
        """
        This class offers some useful functions for a better SpaCy experience.

        Parameters
        ----------
        model_type: str
            The name of the model which SpaCy should load.
            You should have downloaded it beforehand.
            As of SpaCy 2.0 that would be 'en', 'en_core_web_sm',
            'en_core_web_md' and 'en_core_web_lg'
        pipeline_to_remove: list of str
            The names of the pipeline steps to disable.
            Keep the dependencies of the parts in mind!
        """
        self._model_type = model_type
        self._pipeline_to_remove = pipeline_to_remove
        self.nlp = None

    def get_nlp(self):
        """
        Returns
        -------
        spacy.lang.en.English
            A SpaCy language model that can do
            operations on the english language
        """
        if self.nlp is None:
            # spacy.prefer_gpu()
            self.nlp = spacy.load(self._model_type, disable=self._pipeline_to_remove)
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        return self.nlp
