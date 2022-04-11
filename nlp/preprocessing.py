# -*- coding: utf-8 -*-

# from nlp.functions.utils import appos_dict, slangs_dict, stop_words_list, emo
from .utils import appos_dict, slangs_dict, emoticons_dict

from typing import List, Collection, Union
import re
import multiprocessing as mp
from tqdm.auto import tqdm
import copy
import pickle

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token, DocBin


_SPACY_LM = "en_core_web_sm"
_ACTIONS = ['appos', 'slang', 'emoticons', 'emoticons_del', 'repeated_chars',
            'sep_digit_text', 'eol_remove', 'eol_replace', 'html',
            'punct', 'extra_space', 'email', 'url', 'stop_words', 'proper_noun', 'phone_number',
            'number', 'single_char', 'lemmas']


class _CustomExtension:
    """
    This class marks tokens for removal based on conditions
    """
    def __init__(self, nlp: Language):
        """
        Class constructor that defines the new extensions to add in the Spacy Pipeline.

        Args:
            nlp: Spacy language model
        """

        if not Token.has_extension("proper_noun"):
            Token.set_extension("proper_noun", default=False)

        if not Token.has_extension("single_char"):
            Token.set_extension("single_char", default=False)

        if not Token.has_extension("phone_number"):
            Token.set_extension("phone_number", default=False)

    def __call__(self, doc: Doc) -> Doc:
        """
        Define how the custom extensions are calculated.

        Args:
            doc (Doc): Spacy document type.

        Returns:
            Doc: The spacy document Doc including the new extensions.
        """

        for t in doc:
            if t.ent_type_ == 'PERSON':  # Proper nouns
                t._.proper_noun = True
            if len(t.text) < 2:  # single char words
                t._.single_char = True
            if t.like_num and len(t.text) >= 9:  # phone number
                t._.phone_number = True

        # Return a new Doc
        return doc


@Language.factory("custom_extension")
def _custom_extension(nlp: Language, name: str):
    return _CustomExtension(nlp)


class Pipeline:
    """
    This class wraps a spacy language model with some other features.
    There are two main preprocessing actions:
    - Irreversible actions: Actions that modify the source texts, not covered by in spacy pipeline
    - Reversible actions: Actions that are performed on a lazy way and do not modify source (spacy)

    This class adds standard Python capabilities such as list indexing and slicing.
    Also supports multiprocessing.
    """

    def __init__(self,
                 config: dict,
                 spacy_language: str = _SPACY_LM,
                 exclude: List[str] = [],
                 keep: List[str] = [],
                 n_process: int = 1,
                 progress_bar: bool = False):
        """
        Class constructor.

        Args:
            config (dict): configuration dict with attributes to enable/disable in the Pipeline execution
            spacy_language (str): language attribute for spacy model "en" for english
            exclude (List[str]): Names of pipeline components to exclude. Excluded components won’t be loaded.
            keep (List[str]): Names of pipeline components to include. Keep list has preference over exclude
            n_process (int): The number of process to run in parallel for multi CPU processing
            progress_bar (bool):  If True enable progress on screen progress bar
        """

        self.progress_bar = progress_bar

        self._config = config.copy()
        # Make sure all possible actions exist in config
        for action in _ACTIONS:
            if action not in self.config:
                self.config[action] = False

        # Init docs list and number of processes
        self.docs = []
        self.n_process = mp.cpu_count() if n_process == -1 else n_process

        # Align 'keep' list and 'exclude' list. Keep list has preference over exclude
        exclude_list = exclude.copy()
        exclude_list = [action for action in exclude_list if action not in keep]
        self.exclude = exclude_list

        # Create the Language model
        self.language = spacy_language if spacy_language is not None else _SPACY_LM
        self.nlp = self.get_spacy(self.language, self.exclude)

        # Replacement functions init
        self.appos_dict = appos_dict
        self.slangs_dict = slangs_dict
        self.emoticons_dict = emoticons_dict
        self.emoticons_dict = {k: '' for k in emoticons_dict.keys()} if self._config['emoticons_del'] \
            else self.emoticons_dict
        self.re_rep_chars = re.compile(r'(.)\1+')
        self.re_sep_digit = re.compile(r'([\d]+)([a-zA-Z]+)')
        self.re_eol = re.compile('\\r\\n|\\n|\\r')
        self.re_html = re.compile(r'<.*?>')

    @staticmethod
    def get_spacy(language: str = _SPACY_LM,
                  exclude: list = []
                  ) -> spacy.lang:
        """
        NLP spacy model getter.

        Args:
            language (str): Language Spacy model, this model will be loaded from disc.
            exclude (list): Names of pipeline components to exclude. Excluded components won’t be loaded.

        Returns:
            spacy.lang: NLP Spacy model.
        """
        nlp = spacy.load(language, exclude=exclude)
        nlp.add_pipe("custom_extension", last=True)
        return nlp

    def save(self,
             path: str
             ) -> None:
        """
        Save NLP pipeline model in disk.

        Args:
            path(str): Pipeline model path where the model will be saved.

        Returns: None
        """
        copied_object = copy.copy(self)
        copied_object.nlp = copied_object.nlp.to_bytes()
        copied_object.docs = DocBin(docs=self.docs, store_user_data=True).to_bytes()
        pickle.dump(copied_object, open(path, "wb"))

    @classmethod
    def load(cls,
             path: str
             ) -> spacy.lang:
        """
        Load NLP pipeline model from disk

        Args:
            path (str): Path of pipeline model to load.

        Returns:
            spacy.lang: NLP spacy model
        """
        # Load serialized object
        pp = pickle.load(open(path, "rb"))

        # Load spacy LM
        nlp = cls.get_spacy(language=pp.language, exclude=pp.exclude)
        pp.nlp = nlp.from_bytes(pp.nlp)

        # Load docs
        doc_bin = DocBin().from_bytes(pp.docs)
        pp.docs = list(doc_bin.get_docs(pp.nlp.vocab))

        return pp

    def __getitem__(self,
                    key: Union[int, str] = 7
                    ) -> Doc:
        """
        Get the key item from the class iterator .docs

        Args:
            key (Union[int,str], optional): Indicate the index position to get.

        Raises:
            KeyError: If key is not in the .docs iterator will raise an Error.

        Returns:
            Doc: Spacy document.
        """

        if len(self.docs) == 0:
            raise KeyError('Call fit method first')

        # If indexing is slice, then return a view of this object
        if isinstance(key, slice):
            view = Pipeline(self.config,
                            n_process=self.n_process,
                            progress_bar=self.progress_bar)
            view.nlp = self.nlp
            view.docs = self.docs[key]
            return view

        return self.docs[key]

    def __len__(self) -> int:
        """
        Set the len method according the number of processed documents.

        Returns:
            int: processed documents length

        """
        return len(self.docs)

    def __str__(self) -> str:
        """
        Set the class string type name

        Returns:
            str: class string type name

        """
        return str(self.nlp.pipeline)

    def _replace_lookups(self,
                         text: str
                         ) -> str:
        """
        This functions runs text cleaning actions on the raw text before passing it to spacy.
        Actions are:
        - Appos replacement.
        - Slang replacement.
        - Emoticons replacement/removal.
        - Repeated characters removal.
        - Separate digits from words.

        Args:
            text (str): text to be processed.

        Returns:
            str: The cleaned text in string format.
        """

        # Remove or replace eol symbols such as \r and \n
        if self.eol_remove:
            text = self.re_eol.sub('', text)
        elif self.eol_replace:
            text = self.re_eol.sub('. ', text)

        # Remove HTML
        if self.html:
            text = self.re_html.sub('', text)

        # Clean repeated characters to a maximum of 2
        if self.repeated_chars:
            text = self.re_rep_chars.sub(r'\1\1', text)

        # Separate digit from text such as 3lines -> 3 lines
        if self.sep_digit_text:
            text = self.re_sep_digit.sub(r'\1 \2', text)

        # Run replacements
        words = text.split()
        if self.appos:
            words = [self.appos_dict[w] if w in self.appos_dict else w for w in words]

        if self.slang:
            words = [self.slangs_dict[w] if w in self.slangs_dict else w for w in words]

        if self.emoticons or self.emoticons_del:
            words = [self.emoticons_dict[w] if w in self.emoticons_dict else w for w in words]

        return ' '.join(words)

    def _clean(self,
               texts: Collection[str]
               ) -> List[str]:
        """
        Performs a cleaning implementing the wanted replace_lookups in a set of 'text' documents.

        Args:
            texts (Collection[str]): text documents to process.

        Returns:
            Collections[str]: cleaned text documents.
        """
        # Run look up replacements only if needed
        if any([self.appos, self.slang, self.emoticons, self.repeated_chars,
                self.sep_digit_text, self.eol_remove, self.eol_replace, self.html]):
            if self.n_process == 1:
                texts = [self._replace_lookups(t) for t in tqdm(texts, disable=not self.progress_bar,
                                                                desc='Cleaning Data')]
            else:
                with mp.Pool(processes=self.n_process) as pool:
                    texts = pool.map(self._replace_lookups, texts)
        return texts

    def fit(self,
            texts: Collection[str]):
        """
        Runs spacy language model on a list of texts.
        It performs 2 steps:
            - A destructive transformation (meaning that can't be reversed).
            - Spacy preprocessing.

        A list of Docs is stored internally.

        Args:
            texts (Collection[str]): A list, array or Series of texts.

        Returns:
            Preprocessor: This object.
        """
        # Run look up replacements only if needed
        texts = self._clean(texts)

        # Generate docs
        self.docs = list(tqdm(self.nlp.pipe(texts, n_process=self.n_process),
                              disable=not self.progress_bar,
                              total=len(texts),
                              desc='Running Spacy'))

        return self

    def _mask(self,
              token: Token
              ) -> Union[Token, str, bool]:
        """
        Returns a mask if the token meets the requirements and the mask was configured.

        Args:
            token: Spacy Token type that contains the token extensions (Default extensions and custom).

        Returns:
            Union[Token, str]: If mask was configured will return a string otherwise return the token input.
        """
        if isinstance(self._config['email'], str) and token.like_email:
            return self._config['email']

        if isinstance(self._config['url'], str) and token.like_url:
            return self._config['url']

        if isinstance(self._config['phone_number'], str) and token._.phone_number:
            return self._config['phone_number']

        if isinstance(self._config['number'], str) and token.like_num:
            return self._config['number']

        if isinstance(self._config['proper_noun'], str) and token._.proper_noun:
            return self._config['proper_noun']

        return token

    def _filter(self,
                doc: Doc
                ) -> List[str]:
        """
        Tries to apply masks (if possible) and filter tokens based on current configuration.

        Args:
            doc (spacy Doc): The spacy Doc object to be processed.

        Returns:
            List[str]: List of filtered strings.
        """
        tokens = []
        for t in doc:

            # Try to apply a mask
            token = self._mask(t)
            if isinstance(token, str):
                tokens.append(token)
                continue

            # Apply filter
            if not (self._config['punct'] and t.is_punct) and \
               not (self._config['extra_space'] and t.is_space) and \
               not (self._config['stop_words'] and t.is_stop) and \
               not (self._config['single_char'] and t._.single_char) and \
               not (self._config['email'] and t.like_email) and \
               not (self._config['url'] and t.like_url) and \
               not (self._config['number'] and t.like_num) and \
               not (self._config['phone_number'] and t._.phone_number) and \
               not (self._config['proper_noun'] and t._.proper_noun):
                tokens.append(t.lemma_ if self._config['lemmas'] else t.text)
        return tokens

    def transform(self) -> List[str]:
        """
        Transform the data into a list of texts. Tokens are extracted based on current configuration.

        Raises:
            KeyError: This method should be called only after calling method 'fit'.

        Returns:
            List[str]: A list of preprocessed texts.
        """
        if len(self.docs) == 0:
            raise KeyError('Call fit method first')

        texts = []
        for doc in self.docs:
            words = self._filter(doc)
            texts.append(Doc(doc.vocab, words=words).text if len(words) > 0 else '')

        texts = [t.replace(" '", "'") for t in texts]  # remove space left before apostrophe

        return texts

    def text(self,
             key: Union[str, int]
             ) -> str:
        """
        Applies preprocessing for a given index. Call this method when you don't want the full corpus returned.

        Args:
            key (Union[str, int]): index.

        Returns:
            str: Returned preprocessed string.
        """
        doc = self[key]
        words = self._filter(doc)
        text = Doc(doc.vocab, words=words).text if len(words) > 0 else ''
        text = text.replace(" '", "'")

        return text

    def tokenize(self) -> List[List[str]]:
        """
        Transform the data into a list of list of tokens. Tokens are extracted based on current configuration.

        Raises:
            KeyError: This method should be called only after calling method 'fit'.

        Returns:
            List[List[Token]]: A list of list of tokens.
        """
        if len(self.docs) == 0:
            raise KeyError('Call fit method first')

        texts = []
        for doc in self.docs:
            words = self._filter(doc)
            texts.append(words)

        return texts

    def fit_transform(self,
                      texts: Collection[str]
                      ) -> List[str]:
        """
        Preprocess data and yield the output texts.
        This method does not store Doc objects in memory, so it is recommended to process large chunks of data without
        running into memory issues.

        Args:
            texts (Collection[str]): A list, array or Series of texts.

        Yields:
            List[str]: A list of preprocessed texts.
        """
        texts = self._clean(texts)
        processed_texts = []

        for doc in tqdm(self.nlp.pipe(texts, n_process=self.n_process),
                        disable=not self.progress_bar,
                        total=len(texts),
                        desc='Running Spacy'):
            words = self._filter(doc)
            processed_texts.append(Doc(doc.vocab, words=words).text if len(words) > 0 else '')

        processed_texts = [t.replace(" '", "'") for t in processed_texts]  # remove space left before apostrophe

        return processed_texts

    @property
    def config(self): return self._config

    @config.setter
    def config(self, value: dict): self._config = value

    @property
    def appos(self): return self._config['appos']

    @appos.setter
    def appos(self, value: bool): self._config['appos'] = value

    @property
    def slang(self): return self._config['slang']

    @slang.setter
    def slang(self, value: bool): self._config['slang'] = value

    @property
    def eol_remove(self): return self._config['eol_remove']

    @eol_remove.setter
    def eol_remove(self, value: bool): self._config['eol_remove'] = value

    @property
    def eol_replace(self): return self._config['eol_replace']

    @eol_replace.setter
    def eol_replace(self, value: bool): self._config['eol_replace'] = value

    @property
    def html(self): return self._config['html']

    @html.setter
    def html(self, value: bool): self._config['html'] = value

    @property
    def sep_digit_text(self): return self._config['sep_digit_text']

    @sep_digit_text.setter
    def sep_digit_text(self, value: bool): self._config['sep_digit_text'] = value

    @property
    def emoticons(self): return self._config['emoticons']

    @emoticons.setter
    def emoticons(self, value: bool): self._config['emoticons'] = value

    @property
    def emoticons_del(self): return self._config['emoticons_del']

    @emoticons_del.setter
    def emoticons_del(self, value: bool): self._config['emoticons_del'] = value

    @property
    def repeated_chars(self): return self._config['repeated_chars']

    @repeated_chars.setter
    def repeated_chars(self, value: bool): self._config['repeated_chars'] = value

    @property
    def punct(self): return self._config['punct']

    @punct.setter
    def punct(self, value: bool): self._config['punct'] = value

    @property
    def extra_space(self): return self._config['extra_space']

    @extra_space.setter
    def extra_space(self, value: bool): self._config['extra_space'] = value

    @property
    def email(self): return self._config['email']

    @email.setter
    def email(self, value: Union[bool, str]): self._config['email'] = value

    @property
    def proper_noun(self): return self._config['proper_noun']

    @proper_noun.setter
    def proper_noun(self, value: Union[bool, str]): self._config['proper_noun'] = value

    @property
    def phone_number(self): return self._config['phone_number']

    @phone_number.setter
    def phone_number(self, value: Union[bool, str]): self._config['phone_number'] = value

    @property
    def stop_words(self): return self._config['stop_words']

    @stop_words.setter
    def stop_words(self, value: bool): self._config['stop_words'] = value

    @property
    def number(self): return self._config['number']

    @number.setter
    def number(self, value: Union[bool, str]): self._config['number'] = value

    @property
    def single_char(self): return self._config['single_char']

    @single_char.setter
    def single_char(self, value: bool): self._config['single_char'] = value

    @property
    def url(self): return self._config['url']

    @url.setter
    def url(self, value: Union[bool, str]): self._config['url'] = value

    @property
    def lemmas(self): return self._config['lemmas']

    @lemmas.setter
    def lemmas(self, value: bool): self._config['lemmas'] = value
