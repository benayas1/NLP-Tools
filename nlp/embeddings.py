from typing import List, Union
import numpy as np
import pandas as pd
import spacy.tokens.doc
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from tqdm.auto import tqdm
from .functions import count_words, count_ngrams
from .preprocessing import Pipeline
from nltk.util import ngrams
import gensim
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import re
from sklearn.base import BaseEstimator
from collections.abc import Iterable

SPACY_DEFAULT = "en_core_web_sm"


class BagOfPOSEmbeddings(BaseEstimator):
    """
    Generates embeddings using a bag of Part-of-Speech.
    POS tags are assigned by spacy.
    The length of the resulting vectors depends on how many different POS were used in the corpus.
    """
    def __init__(self,
                 part_of_speech: str = 'tag',
                 progress_bar: bool = False
                 ):
        """
        Class constructor

        Args:
            part_of_speech (str, optional): Possible values are 'tag' or 'pos'. It indicates which spacy tagging to use.
            'tag' for dependency tags and 'pos' for part-of-speech tags.
            progress_bar (bool, optional): Whether to display a progress bar or not. Defaults to False.

        Raises:
            ValueError: If part_of_speech is not 'tag' or 'pos'.
        """

        if part_of_speech not in ['tag', 'pos']:
            raise ValueError('part_of_speech must be either tag or pos')

        self.part_of_speech = part_of_speech
        self.progress_bar = progress_bar
        self.feature_names = []

    # todo Merge: to keep same transformation vector order when calling class instance multiple times
    def _set_feature_names(self,
                           df_tags: pd.DataFrame
                           ) -> None:
        """
        Set the parameter self.feature_names. This parameter must be set with the first call
        to the 'fit_transform' method. The next calls will be use this parameter to sort the embeddings
        in the same way, considering always the same column order when next calls to 'fit_transform' method.

        Args:
            df_tags (pd.DataFrame): Tags data where rows represent documents and columns the tokens tags.

        Returns:
            None.
        """
        self.feature_names = list(df_tags.columns)

    def _sort_embeddings(self,
                         df_tags: pd.DataFrame
                         ) -> pd.DataFrame:
        """
        Return a sorted 'df_tags' dataframe if 'self.feature_names' has been set, otherwise will return the same
        dataframe setting the parameter 'self,features_names'.

        Args:
            df_tags (pd.DataFrame): Tags data where rows represent documents and columns tokens frequency.

        Returns:
            pd.DataFrame: Returns the same input data 'df_tags' if 'self.feature_names' is empty, otherwise
            return the 'df_tags' considering the columns and columns order in 'self.feature_names'.
        """
        if self.feature_names:
            df_tags_sorted = pd.DataFrame(columns=self.feature_names)
            for column in self.feature_names:
                if column in df_tags.columns:
                    df_tags_sorted[column] = df_tags[column]
                else:
                    df_tags_sorted[column] = [0.0] * len(df_tags)

            return df_tags_sorted

        else:
            self._set_feature_names(df_tags)
            return df_tags

    def fit_transform(self,
                      X: Union[Pipeline, List[spacy.tokens.doc.Doc], List[str]]
                      ) -> np.array:
        """
        Generates the embedding matrix by counting the tag frequency in the 'X' documents.

        Args:
            X (Union[Pipeline, List[spacy.tokens.doc.Doc]]): Data to be processed, contains the text documents to
            process.

        Returns:
            np.ndarray: The calculated pos tag embeddings with shape (n_samples, n_dimensions).
        """
        if not isinstance(X, Pipeline):
            tagger = spacy.load(SPACY_DEFAULT, exclude=['ner', 'lemmatizer', 'parser'])

        tag_vectors = []
        for d in tqdm(X, disable=not self.progress_bar, desc='Generating BOW'):
            # If input is an iterable of texts, we need to get the Tokens with tags
            if isinstance(d, str):
                d = tagger(d)
            tmp_vector = {}
            for token in d:
                tag = token.tag_ if self.part_of_speech == 'tag' else token.pos_
                if tag not in tmp_vector:
                    tmp_vector[tag] = 1
                else:
                    tmp_vector[tag] += 1
            tag_vectors.append(tmp_vector)

        # We need to normalize the vectors since long sentence might have larger values
        # Normalize using L2 norm
        # create fit method or equivalent to keep the same tags order in the tag_vector df-array
        #  otherwise when calling transform method several times will return an array with different tag order
        #  another possibility is to used order using spacy glosary, but than involves a lot of empty tags in the
        #  matrix --> https://github.com/explosion/spaCy/blob/master/spacy/glossary.py (dict do not guarantee key order)
        # tag_vectors = pd.DataFrame(tag_vectors).fillna(0).values
        tag_vectors = pd.DataFrame(tag_vectors).fillna(0)
        tag_vectors = self._sort_embeddings(df_tags=tag_vectors)
        tag_vectors = tag_vectors.values
        tag_vectors = tag_vectors/np.linalg.norm(tag_vectors, 2, axis=1).reshape(-1, 1)
        tag_vectors = np.nan_to_num(tag_vectors, nan=0).astype(np.float32)
        return tag_vectors


class KeyWordsEmbeddings(BaseEstimator):

    """
    Generate embeddings counting the frequency of keyword in the documents.
    Keywords are considered unique words, bigrams, trigrams and 4-grams.
    It is recommended to clean up the corpus first, to remove stop words and other common words.
    """
    def __init__(self,
                 top_k: int = 20,
                 mode: str = 'count',
                 key_words: List[str] = None,
                 progress_bar: bool = False
                 ):
        """
        Class constructor.

        Args:
            top_k (int): Top most commons words, bi-grams and tri-grams to consider.
            mode (str): Mode to compute BOW word scores, 'count' computes the frequency -'binary' transforms it to bool.
            key_words (List[str]): Force to keep those phrase words in the top_k selection.
            progress_bar (bool):  Whether to display a progress bar or not. Defaults to False.
        """

        self.top_k = top_k  # top most commons words, bigrams, trigrams, etc.
        self.mode = mode
        self.key_words = key_words if key_words is not None else []
        self.progress_bar = progress_bar
        self.top_keyword_vocab = {}

    def fit(self,
            X: Union[Pipeline, List[str]]
            ):
        """
        Creates the keyword vocabulary based on input data. The vocabulary will include the 'top_k' most frequently
        keywords + the custom 'self.key_words' keywords.

        Args:
            X (Union[Pipeline, List[spacy.tokens.doc.Doc]]): Data to be processed, contains the text documents to
            process.

        Returns:
            self: Pointer to self.
        """
        keyword_vocab = set()

        if isinstance(X, Pipeline):
            corpus = ' '.join(X.transform()).lower()
        else:
            corpus = ' '.join(X).lower()

        # Single words: exclude keywords to include those words later in transform method. If keyword is in the top_k
        # it is replaced by another candidate in the index list
        df_count_words = pd.DataFrame.from_dict(count_words(corpus), orient='index', columns=['Count']) \
                                     .sort_values(by='Count', ascending=False)
        candidates = [w for w in df_count_words.iloc[:self.top_k + len(self.key_words)].index
                      if w not in self.key_words][:self.top_k]
        keyword_vocab.update(candidates)

        for n in [2, 3, 4]:
            key_word_ngrams = [list(ngram)[0] for ngram in [count_ngrams(key_word, n) for key_word in self.key_words]
                               if ngram]
            df_ngrams = pd.DataFrame.from_dict(count_ngrams(corpus, n), orient='index', columns=['Count']) \
                                    .sort_values(by='Count', ascending=False)
            candidates = [w for w in df_ngrams.iloc[:self.top_k + len(self.key_words)].index
                          if w not in key_word_ngrams][:self.top_k]
            keyword_vocab.update(candidates)

        # The dict index will be assigned in the same order of dict updates 1gram, 2gram etc.
        self.top_keyword_vocab = {k: i for i, k in enumerate(keyword_vocab)}
        return self

    def transform(self,
                  X: Union[Pipeline, List[str]]
                  ) -> np.array:
        """
        Generates the embedding matrix using the precalculated vocabulary.

        Args:
            X (Union[Pipeline, List[str]]): Data to be processed, contains the text documents to process.

        Returns:
            np.ndarray: The calculated embeddings with shape (n_samples, n_dimensions).
        """

        if not hasattr(self, 'top_keyword_vocab'):
            raise KeyError('Call fit method first')

        if isinstance(X, Pipeline):
            texts = X.transform()
        else:
            texts = X

        # Create a BoW embeddings
        kw_embeddings = np.zeros((len(texts), len(self.top_keyword_vocab)+len(self.key_words))).astype(np.float32)
        for i, t in enumerate(tqdm(texts, disable=not self.progress_bar, desc='Creating Embeddings')):
            t = t.lower()

            for j, w in enumerate(self.key_words):
                is_composed_word = True if len(w.split()) > 1 else False
                if is_composed_word:
                    kw_embeddings[i, j+len(self.top_keyword_vocab)] += len(re.findall(w, t))
                else:
                    kw_embeddings[i, j + len(self.top_keyword_vocab)] += count_words(t).get(w, 0)

            # Count 1-gram words
            for w in t.lower().split():
                if w in self.top_keyword_vocab:
                    kw_embeddings[i, self.top_keyword_vocab[w]] += 1

            # Count n-gram words
            for n in [2, 3, 4]:
                for w in ngrams(t.split(), n):
                    if w in self.top_keyword_vocab:
                        kw_embeddings[i, self.top_keyword_vocab[w]] += 1

        if self.mode == 'count':
            kw_embeddings = kw_embeddings/kw_embeddings.max()  # Normalize vectors to [0,1]
        if self.mode == 'binary':
            kw_embeddings[kw_embeddings > 0] = 1

        return kw_embeddings

    def fit_transform(self,
                      X: Union[Pipeline, List[str]]
                      ) -> np.array:
        """
        Performs fit and transform method together.

        Args:
            X (Union[Pipeline, List[str]]): Data to be processed, contains the text documents to process.

        Returns:
            np.ndarray: The calculated embeddings with shape (n_samples, n_dimensions).
        """
        return self.fit(X).transform(X)


class LdaEmbeddings(BaseEstimator):
    """
    Generates embeddings using the topic extraction algorithm LDA.
    """
    def __init__(self,
                 n_topics: int = 20,
                 n_process: int = 1,
                 progress_bar: bool = False
                 ):
        """
        Class constructor.

        Args:
            n_topics (int): Amount of topics to train.
            n_process (int): Number of CPU's to use for parallel computing.
            progress_bar (bool):  Whether to display a progress bar or not. Defaults to False.
        """

        self.n_topics = n_topics
        self.n_process = n_process
        self.progress_bar = progress_bar
        self.lda_model_tfidf = None
        self.dictionary = None

    @staticmethod
    def _get_tokens(X: Union[Pipeline, List[str]]
                    ) -> List[List[str]]:
        """
        Get text tokens from X, where the text is tokenized at word level. If text
        is already tokenized return same text.

        Args:
            X (Union[Pipeline, List[List[str]]]): Instance of fitted 'Pipeline' that contains a set of text documents,
            or an iterable of list of word tokens.

        Returns:
             List[List[str]]: Where each list of list is a document of word tokens (tokenized docs).
        """
        if isinstance(X, Pipeline):
            processed_docs = X.tokenize()
        else:
            if isinstance(X, Iterable) and isinstance(X[0], str):
                tokenizer = Tokenizer(English().vocab)
                processed_docs = [[w.text for w in tokenizer(text)] for text in X]
            else:
                processed_docs = X

        return processed_docs

    # @todo Merge: code used by .fit() and .fit_transform() method. Idea is transform method apply the same model
    #   instead generate a new one with each call
    def _build_corpus(self,
                      X: Union[Pipeline, List[str]]
                      ) -> gensim.interfaces.TransformedCorpus:
        """
        Calculate the bag of words corpus using the TFIDF approach.

        Args:
            X (Union[Pipeline, List[str]]): Instance of fitted 'Pipeline' that contains a set of text documents
            or an iterable of list of word tokens.

        Returns:
            gensim.interfaces.TransformedCorpus: That contains the TFIDF matrix for each document in X.
        """

        processed_docs = self._get_tokens(X)

        # First generate the bag of words of every utterance
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in tqdm(processed_docs, desc='Generating BOW')]

        # Create TF-IDF Matrix and corpus
        tfidf = gensim.models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]

        assert len(bow_corpus) == len(corpus_tfidf)
        return corpus_tfidf

    def fit(self,
            X: Union[Pipeline, List[str]]
            ):
        """
        Set up the LDA model components using the text docs in X.

        Args:
            X (Union[Pipeline, List[List[str]]]): Instance of fitted 'Pipeline' that contains a set of text documents
            or an iterable of list of word tokens.

        Returns:
            self: Pointer to self.
        """

        processed_docs = self._get_tokens(X)

        # Generating vocabulary
        dictionary = gensim.corpora.Dictionary(tqdm(processed_docs, desc='Building dictionary'))
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.dictionary = dictionary

        corpus_tfidf = self._build_corpus(X)

        # todo Merge:  transform generated a different topic model each call I moved this from transform() to here
        # TF-IDF
        if self.n_process == 0 or self.n_process == 1:
            lda_model_tfidf = gensim.models.ldamodel.LdaModel(tqdm(corpus_tfidf, desc='Running LDA'),
                                                              num_topics=self.n_topics,
                                                              id2word=self.dictionary,
                                                              passes=2,
                                                              random_state=10)
        else:
            lda_model_tfidf = gensim.models.LdaMulticore(tqdm(corpus_tfidf, desc='Running Multicore LDA'),
                                                         num_topics=self.n_topics,
                                                         id2word=self.dictionary,
                                                         passes=2,
                                                         workers=self.n_process,
                                                         random_state=10)
        self.lda_model_tfidf = lda_model_tfidf

        return self

    def transform(self,
                  X: Union[Pipeline, List[str]]
                  ) -> np.array:
        """
        Perform inference of LDA model 'self.lda_model_tfidf' using the vocabulary from 'self.dictionary' in  the X text
        documents to calculate the embedding topic matrix.

        Args:
            X (Union[Pipeline, List[List[str]]]): Instance of fitted 'Pipeline' that contains a set of text documents
            or an iterable of list of word tokens.

        Returns:
            np.array: The calculated topic embeddings with shape (n_samples, n_topics).
        """

        # if not hasattr(self, 'dictionary'):
        if self.dictionary is None or self.lda_model_tfidf is None:
            raise KeyError('Call fit method first')

        corpus_tfidf = self._build_corpus(X)

        lda_embeddings = np.zeros((len(corpus_tfidf), self.n_topics))

        # We have to do it in this way as not all vectors have equal length
        for i in tqdm(range(len(corpus_tfidf)), desc='Creating Embeddings'):
            c = corpus_tfidf[i]
            vector = self.lda_model_tfidf[c]
            for v in vector:
                lda_embeddings[i, v[0]] = v[1]

        return lda_embeddings

    def fit_transform(self,
                      X: Union[Pipeline, List[str]]
                      ) -> np.array:
        """
        Performs a fit-transform operation building the LDA model and performing model inference in the X text
        documents.

        Args:
            X (Union[Pipeline, List[List[str]]]): Instance of fitted 'Pipeline' that contains a set of text documents
            or an iterable of list of word tokens.

        Returns:
            np.array: The calculated topic embeddings with shape (n_samples, n_topics).
        """
        processed_docs = self._get_tokens(X)
        return self.fit(processed_docs).transform(processed_docs)


class SentenceTransformerEmbeddings(BaseEstimator):
    """
    Generates embeddings using SentenceTransformers.
    SentenceTransformer is an open source library that uses Transformers to produce sentence embeddings.
    This class is a wrapper around it.
    """

    def __init__(self,
                 model_string: str,
                 device: str = 'cpu',
                 progress_bar: bool = False
                 ):
        """
        Class constructor.

        Args:
            model_string (str): model name (to download model from transformers repo) or model path (from local path).
            device (str): device type cpu or gpu(cuda:0).
            progress_bar (bool): Whether to display a progress bar or not. Defaults to False.
        """
        self.bi_encoder = SentenceTransformer(model_string)
        self.device = device
        self.progress_bar = progress_bar

    def fit_transform(self,
                      X: Union[Pipeline, List[List[str]]],
                      ) -> np.ndarray:
        """
        Performs a model prediction extracting the embeddings output from the given transformer model.

        Args:
            X (Union[Pipeline, List[str]]):  Data to be processed, contains the text documents to process.

        Returns:
            np.ndarray: Embeddings matrix from transformer model with shape (documents, embeddings_size)
        """

        if isinstance(X, Pipeline):
            corpus = X.transform()
        else:
            corpus = X

        embeddings = self.bi_encoder.encode(corpus,
                                            show_progress_bar=self.progress_bar,
                                            device=self.device,
                                            convert_to_numpy=True)
        return embeddings


class TransformerEmbeddings(BaseEstimator):
    """
    Generates embeddings using HuggingFace's models.
    There are 3 different ways of generating sentence embeddings:
    - Average Pooling
    - Max Pooling
    - CLS token
    """

    def __init__(self,
                 model_string: str,
                 pooling: str = 'mean',
                 device: str = 'cpu',
                 batch_size: int = 256,
                 progress_bar: bool = False
                 ):
        """
        Class constructor.

        Args:
            model_string (str):  Model name (to download model from transformers repo) or model path (from local path).
            pooling (str): Function to generate the embedding: 'mean', 'max', 'cls'.
            device (str): Device type cpu or gpu(cuda:0).
            batch_size (int): How many samples per batch to load.
            progress_bar (bool): Whether to display a progress bar or not. Defaults to False.
        """

        if pooling not in ['mean', 'max', 'cls']:
            raise ValueError('pooling must be mean, max or cls')
        self.pooling = pooling
        self.batch_size = batch_size

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.model = AutoModel.from_pretrained(model_string)
        self.progress_bar = progress_bar

    def fit_transform(self,
                      X: Union[Pipeline, List[List[str]]],
                      ) -> np.ndarray:
        """
        Generates a feature matrix using the embeddings layers in the given transformer model. The 'last_hidden_state'
        is summarized using selected pooling method.

        Args:
            X (Union[Pipeline, List[str]]):  Data to be processed, contains the text documents to process.

        Returns:
            np.ndarray: Embeddings matrix from transformer model with shape (documents, embeddings size).
        """

        if isinstance(X, Pipeline):
            corpus = X.transform()
        else:
            corpus = X

        self.model = self.model.to(self.device)
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus), self.batch_size), disable=not self.progress_bar, desc='Generating Embeddings'):
                batch = corpus[i:i+self.batch_size]
                tokens = self.tokenizer(batch,
                                        padding=True,
                                        truncation=True,
                                        max_length=500,
                                        add_special_tokens=True,
                                        return_tensors="pt")
                tokens = tokens.to(self.device)

                output = self.model(**tokens)['last_hidden_state']

                # Embeddings can be average pooling, max pooling or CLS token.
                if self.pooling == 'mean':
                    attention = tokens['attention_mask'].unsqueeze(-1).expand(output.size()).float()
                    sum_embeddings = torch.sum(output * attention, 1)
                    sum_mask = torch.clamp(attention.sum(1), min=1e-9)
                    output = sum_embeddings / sum_mask
                    output = output.cpu().numpy()
                if self.pooling == 'max':
                    attention = tokens['attention_mask'].unsqueeze(-1).expand(output.size()).float()
                    output[attention == 0] = -1e9
                    output = output.cpu().numpy().max(axis=1)
                if self.pooling == 'cls':
                    output = output.cpu().numpy()
                    output = output[:, 0, :]

                embeddings.append(output)
        embeddings = np.concatenate(embeddings)
        return embeddings
