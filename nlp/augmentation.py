from abc import ABC, abstractmethod
import warnings
import itertools
import re
import random
from typing import List, Union, Iterable, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from .utils import stop_words_list, QWERTY_dist1_dict
import gensim
import faiss


class Augmenter(ABC):
    """
    Augmenter base class.
    """

    @abstractmethod
    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Raises:
            NotImplementedError: Not implemented in this class.
        """
        raise NotImplementedError('Implement the method')


class Sequence(Augmenter):
    """
    Applies a sequence of augmentations.
    """

    def __init__(self,
                 augmenters: List[Augmenter],
                 p: List[float] = None
                 ):
        """
        Class constructor.

        Args:
            augmenters (List[Augmenter]): List of Augmenters to run.
            p (List[float], optional): Probabilities  of each Augmenter to run. If None it will assign 100%
            probabilities to each Augmenter.
        """
        self.augmenters = augmenters
        if p is None:
            self.p = [1] * len(augmenters)
        else:
            assert len(augmenters) == len(p)
            self.p = p

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        probs = np.random.rand(len(self.augmenters))
        for i, p in enumerate(probs):
            if p <= self.p[i]:
                texts = self.augmenters[i].augment(texts)
        return texts


class OneOf(Augmenter):
    """
    Applies just one augmentation out of the list of the Augmenters.
    """

    def __init__(self,
                 augmenters: List[Augmenter],
                 p: List[float] = None
                 ):
        """
        Class constructor.

        Args:
            augmenters (List[Augmenter]): List of Augmenters to run.
            p (List[float], optional): Probabilities for each Augmenter. It has to be length N_Augmenters - 1.
            Defaults to None.
        """
        self.augmenters = augmenters
        if p is None:
            self.p = [1 / len(augmenters)] * len(augmenters)
        else:
            assert len(augmenters) == len(p) + 1
            # assert sum(augmenters) <= 1.0
            self.p = p
            self.p.append(1 - sum(p))

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        if not isinstance(texts, np.ndarray):
            texts = np.array(texts)

        idx = list(range(len(self.augmenters)))
        # For each text sample indicate witch augmentation must be performed
        choices = np.random.choice(idx, size=len(texts), replace=True, p=self.p)

        uniques = np.unique(choices)

        for i in uniques:
            # Performs the i augmentation, overwritten text[i] with their corresponding transformation
            texts[choices == i] = self.augmenters[i].augment(texts[choices == i])

        return texts


class BackTranslation(Augmenter):
    """
    Applies back translation on a list of texts.
    """

    def __init__(self,
                 language: str = 'de',
                 models_path: str = None

                 ):
        """
        Class constructor.

        Args:
            language (str): translation language.
        """
        self.src = 'en'
        self.tmp = language
        self.device = BackTranslation.get_device()
        self.common_path = 'Helsinki-NLP' if models_path is None else models_path
        self.model_path_src2tmp = f'{self.common_path}/opus-mt-{self.src}-{self.tmp}'
        self.model_path_tmp2src = f'{self.common_path}/opus-mt-{self.tmp}-{self.src}'
        self.model_src2tmp, self.tokenizer_src2tmp = self.download(self.model_path_src2tmp)
        self.model_tmp2src, self.tokenizer_tmp2src = self.download(self.model_path_tmp2src)

        self.model_src2tmp.to(self.device)
        self.model_tmp2src.to(self.device)

        self.model_src2tmp.eval()
        self.model_tmp2src.eval()

    # Helper function to download data for a language
    @classmethod
    def download(cls,
                 model_name: str
                 ) -> Tuple[MarianMTModel, MarianTokenizer]:
        """
        Downloads the required MarianMTModel and tokenizer.

        Args:
            model_name (str): model string to download from HuggingFace.

        Returns:
            MariamMTModel, Tokenizer: model and tokenizer.
        """
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @classmethod
    def get_device(cls) -> str:
        """
        Get the device type.

        Returns:
                str: device type.
        """
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def translate(self,
                  texts: List[str],
                  model: MarianMTModel,
                  tokenizer: MarianTokenizer,
                  language: str
                  ) -> List[str]:
        """
        Translate texts to a target language.

        Args:
            texts (List of str): Texts to be translated.
            model (MarianMTModel): Model to be used.
            tokenizer (MarianTokenizer): Tokenizer object to be used.
            language (str): Target language to translate to.

        Returns:
            List of str: List of translated texts.
        """
        # Format the text as expected by the model
        original_texts = [f"{txt}" if language == "en" else f">>{language}<< {txt}" for txt in texts]

        #   the if condition was removed to avoid filtering documents with unique words (in cause errors when calling)
        #   high level methods like OneOf.augment(). Looks that unique and empty sentences do not broke the execution
        #   the only aspect to consider is when len(s.split(' ')) == 0, the default translation is  "I don't know."
        # original_texts = [s for s in original_texts if len(s.split(' ')) > 1]
        for index, sentence in enumerate(original_texts):
            sentence_size = len(sentence.split(' '))
            if sentence_size <= 1:
                warnings.warn(f"Text: {sentence} of index: {index} has less than one token")

        if len(original_texts) == 0:
            return original_texts

        # Tokenize (text to tokens)
        # @todo merge: deprecated call to transformer tokenizer, and add padding-truncation avoid size errors
        # tokens = tokenizer.prepare_seq2seq_batch(original_texts, return_tensors='pt').to(self.device)
        tokens = tokenizer(original_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Translate
        translated = model.generate(**tokens)

        # Decode (tokens to text)
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        # Translate from source to target language
        translated = self.translate(texts, self.model_src2tmp, self.tokenizer_src2tmp, self.tmp)

        # Translate from target language back to source language
        if len(translated) == 0:
            return translated

        back_translated = self.translate(translated, self.model_tmp2src, self.tokenizer_tmp2src, self.src)

        return back_translated


class SynonymReplacementFast(Augmenter):
    """
    Applies synonyms replacement on a list of texts.
    """

    def __init__(self,
                 data: Union[gensim.models.KeyedVectors, List[str]] = None,
                 device: str = 'gpu',
                 top_k: int = 4,
                 size: int = 300,
                 aug_p: float = 0.1,
                 stop_words: List[str] = stop_words_list
                 ):
        """
        Class constructor

        Args:
            data (Union[gensim.models.KeyedVectors, List[str]]): It has to be a KeyedVectors model (pretrained model),
                string (path) or a list of texts to train a W2V model. Defaults to None.
            device (str, optional): Device type, options are 'gpu'/'cuda' or 'cpu'.
            top_k (int, optional): Neighbors to consider in the clustering to compute synonyms.
            size (int, optional):  Length for embeddings vectors.
            aug_p (float, optional): Synonym replacement probability.
            stop_words (List[str], optional): Optional stop words list.
        """

        # Train W2V model if needed
        if isinstance(data, gensim.models.KeyedVectors):
            self.w2v = data
        elif isinstance(data, str):
            self.w2v = gensim.models.KeyedVectors.load(data, mmap='r')
        else:
            # Training a W2V model
            print(f'Training a W2V model on {len(data)} texts')
            data = [
                [x for x in re.sub(r'[\d]*', '', re.sub(r'[,!?;-]+', '', str(s).lower().replace('.', ''))).split(' ') if
                 x != ''] for s in data]
            model = gensim.models.Word2Vec(sentences=data, vector_size=size, window=10, min_count=1, workers=4)
            self.w2v = model.wv
            print('Training completed')

        # Extract vectors and mappings
        self.word2idx = {}
        self.idx2word = {}
        features = np.zeros(shape=(len(self.w2v.key_to_index), size))
        for i, k in enumerate(self.w2v.key_to_index):
            features[i] = self.w2v[k]
            self.word2idx[k] = i
            self.idx2word[i] = k
        features = features.astype(np.float32)

        # Build network
        self.device = device
        if self.device == 'gpu' or 'cuda' in self.device:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = int(torch.cuda.device_count()) - 1
            self.knn = faiss.GpuIndexFlatIP(res, features.shape[1], flat_config)
            # @todo merge: Below lines was moved out of the IF to re use in the ELSE statement
            # faiss.normalize_L2(features)
            # self.knn.add(features)
        else:
            #  @todo merge:
            # features = features / np.linalg.norm(features, ord=2, axis=1).reshape(-1, 1)
            # self.knn = NearestNeighbors(n_neighbors=top_k).fit(features)
            self.knn = faiss.IndexFlatIP(features.shape[1])

        #  @todo merge:
        faiss.normalize_L2(features)
        self.knn.add(features)
        #  # todo ask: whats about using a odd number?
        self.top_k = 4
        self.aug_p = aug_p
        self.stop_words = stop_words

    def _neighbors(self,
                   vectors: np.array
                   ) -> np.array:
        """
        Method to look for nearest neighbors.

        Args:
            vectors (np.array): Array with embedding vectors where each row represent a document.

        Returns:
            Array of nearest neighbors index.
        """
        # @todo: Perform the neighbors extraction in same way from CPU or GPU.
        faiss.normalize_L2(vectors)
        _, X = self.knn.search(vectors, self.top_k)
        # if self.device == 'gpu' or 'cuda' in self.device:
        #     faiss.normalize_L2(vectors)
        #     _, I = self.knn.search(vectors, self.top_k)
        # else:
        #     vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(-1, 1)
        #     I = self.knn.kneighbors(vectors, n_neighbors=self.top_k, return_distance=False)
        return X

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        # todo merge: delete below parameter, it is set twice
        # new_texts = []

        # Get words to replace
        # Pass from a list of texts to a list of list of tokens
        tokens = [np.array(list(gensim.utils.tokenize(t, lowercase=True))) for t in texts]

        # Get the indices to be replaced by their synonyms
        replace_idx = [
            np.argwhere((np.random.rand(len(t)) < self.aug_p) * (np.array([w not in self.stop_words for w in t])))[:, 0]
            for t in tokens]
        replace_idx = [
            (np.random.randint(len(tokens[i]), size=(1,)) if len(tokens[i]) > 0 else []) if len(r) == 0 else r for i, r
            in enumerate(replace_idx)]

        # Get the words to be replaced based on the indices. Put all the words in a single array
        replace_words = [tokens[i][r] for i, r in enumerate(replace_idx) if len(r) > 0]
        replace_words = np.concatenate(replace_words)

        # Get a list of embeddings vector, one per word to be replaced
        replace_vectors = np.stack(
            [self.w2v[w] if w in self.w2v else np.random.rand(self.w2v.vector_size) for w in replace_words])

        # Look for new words using Nearest Neighbors and the embedding vectors
        X = self._neighbors(replace_vectors.astype(np.float32))
        new_words = [self.idx2word[X[i, r]] for i, r in
                     enumerate(np.random.randint(0, high=self.top_k, size=X.shape[0]))]

        # Replace words in the original texts
        offset = 0
        new_texts = []
        for i, w in enumerate(replace_idx):
            x = tokens[i].copy()
            x[w] = new_words[offset:offset + len(w)]
            new_texts.append(' '.join(x))
            offset += len(w)

        return new_texts


class SwitchAugmentation(Augmenter):
    """
    Class to obtain n_mistakes potential misspellings from a batch of text, with
    possibility of one mistake undoing others. If any number of mistakes wants to
    be ensured, make use of the parameter min_mistakes.
    """

    def __init__(self,
                 n_mistakes: int = 1,
                 min_mistakes: int = 0,
                 ):
        """
        Class constructor.

        Args:
            n_mistakes (int): Maximum amount of mistakes to introduce per document.
            min_mistakes (int): Minimum amount of mistakes to introduce per document.
        """
        # Check parameters are coherent
        # "Minimum of  mistakes cannot be greater than the potential mistakes generated"
        assert n_mistakes >= min_mistakes
        self.n_mistakes = n_mistakes
        self.min_mistakes = min_mistakes

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        if self.n_mistakes <= 0:
            return texts
        else:
            # Parameters of the batch
            max_len, min_len = max([len(elem) for elem in texts]), min([len(elem) for elem in texts])
            # Index to be swapped
            idx_list = [random.randint(0, min_len - 2)] + [random.randint(0, max_len - 2) for _ in
                                                           range(self.n_mistakes - 1)]
            # Check there are at least min_mistakes different mistakes
            if len(list(set(idx_list))) < self.min_mistakes:
                idx_list = idx_list + random.sample([i for i in np.arange(max_len) if i not in idx_list],
                                                    self.min_mistakes - len(list(set(idx_list))))
            # Make changes
            matrix = np.array(list(
                list(itertools.chain.from_iterable([list(elem) + [' '] * (max_len - len(elem)) for elem in texts]))),
                              dtype='str').reshape(len(texts), max_len)
            for idx in idx_list:
                matrix = matrix[:, [i for i in range(idx)] + [idx + 1] + [idx] + [i for i in range(idx + 2, max_len) if
                                                                                  max_len > i]]
            # Back to string format
            # @todo merge optional: iterate over sentences instead over sentences and characters
            matrix = matrix.astype(object)
            output = np.sum(matrix, axis=1)
            output = [sentence.rstrip() for sentence in output]
            # output = [''.join(elem).strip() for elem in matrix]
            return output


class ReplaceAugmentation(Augmenter):
    """
    Class to obtain n_mistakes potential misspellings by replacement of words at
    distance one in the keyboard from a batch of text, with possibility of one
    mistake undoing others. If any number of mistakes wants to be ensured, make
    use of the parameter min_mistakes.
    """

    def __init__(self,
                 n_mistakes: int = 1,
                 min_mistakes: int = 0
                 ):
        """
        Class constructor.

        Args:
            n_mistakes (int): Maximum amount of mistakes to introduce per document.
            min_mistakes (int): Minimum amount of mistakes to introduce per document.
        """
        # Check parameters are coherent
        # "Minimum of  mistakes cannot be greater than the potential mistakes generated"
        assert n_mistakes >= min_mistakes
        self.n_mistakes = n_mistakes
        self.min_mistakes = min_mistakes
        self.qwerty_dict = QWERTY_dist1_dict

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """
        if self.n_mistakes <= 0:
            return texts

        # Parameters of the batch
        max_len, min_len = max([len(elem) for elem in texts]), min([len(elem) for elem in texts])
        # Index to be swapped
        idx_list = [random.randint(0, min_len - 2)] + [random.randint(0, max_len - 2) for _ in
                                                       range(self.n_mistakes - 1)]
        # Check there are at least min_mistakes different mistakes
        if len(list(set(idx_list))) < self.min_mistakes:
            idx_list = idx_list + random.sample([i for i in np.arange(max_len) if i not in idx_list],
                                                self.min_mistakes - len(list(set(idx_list))))
        # Make changes
        matrix = np.array(
            list(list(itertools.chain.from_iterable([list(elem) + [' '] * (max_len - len(elem)) for elem in texts]))),
            dtype='str').reshape(len(texts), max_len)
        for idx in idx_list:
            # @todo merge: fast call to python dict to avoid list verification
            matrix[:, idx] = [random.sample(list(self.qwerty_dict.get(char.lower(), char)), 1)[0]
                              for char in matrix[:, idx]]
            # matrix[:,idx] = [random.sample(list(self.qwerty_dict[char.lower()]),1)[0] if char.lower() in self.qwerty_dict.keys() else char for char in matrix[:,idx]]
        output = [''.join(elem).strip() for elem in matrix]
        return output


class InsertAugmentation(Augmenter):
    """
    Class to obtain n_mistakes potential misspellings by insertion of words at
    distance one in the keyboard from a batch of text, with possibility of one
    mistake undoing others. If any number of mistakes wants to be ensured, make
    use of the parameter min_mistakes.
    """

    def __init__(self,
                 n_mistakes: int = 1,
                 min_mistakes: int = 0
                 ):
        """
        Class constructor.

        Args:
            n_mistakes (int): Maximum amount of mistakes to introduce per document.
            min_mistakes (int): Minimum amount of mistakes to introduce per document.
        """
        # Check parameters are coherent
        # "Minimum of  mistakes cannot be greater than the potential mistakes generated"
        assert n_mistakes >= min_mistakes
        self.n_mistakes = n_mistakes
        self.min_mistakes = min_mistakes
        self.qwerty_dict = QWERTY_dist1_dict

    def augment(self,
                texts: Iterable[str]
                ) -> List[str]:
        """
        Applies augmentation to a sequence of texts.

        Args:
            texts (Iterable[str]): Sequence of texts to be augmented.

        Returns:
            List of str: A list of augmented texts.
        """

        if self.n_mistakes <= 0:
            return texts

        # Parameters of the batch
        max_len, min_len = max([len(elem) for elem in texts]), min([len(elem) for elem in texts])
        # Index to be swapped
        idx_list = [random.randint(0, min_len - 2)] + [random.randint(0, max_len - 2) for _ in
                                                       range(self.n_mistakes - 1)]
        # Check there are at least min_mistakes different mistakes
        if len(list(set(idx_list))) < self.min_mistakes:
            idx_list = idx_list + random.sample([i for i in np.arange(max_len) if i not in idx_list],
                                                self.min_mistakes - len(list(set(idx_list))))
        # Make changes
        matrix = np.array(
            list(list(itertools.chain.from_iterable([list(elem) + [' '] * (max_len - len(elem)) for elem in texts]))),
            dtype='str').reshape(len(texts), max_len)
        i = 0
        for idx in idx_list:
            # @todo merge: fast call to python dict to avoid list verification
            new_column = [random.sample(list(self.qwerty_dict.get(char.lower(), char)), 1)[0]
                          for char in matrix[:, idx + i]]
            # new_column = [random.sample(list(self.qwerty_dict[char.lower()]),1)[0] if char.lower() in self.qwerty_dict.keys() else char for char in matrix[:,idx+i]]
            matrix = np.insert(matrix, idx + i, new_column, axis=1)
            i += 1
        output = [''.join(elem).strip() for elem in matrix]
        return output


def augment_intent(df: pd.DataFrame,
                   augmenter: Augmenter,
                   lower_bound: int = 0,
                   upper_bound: int = np.iinfo(np.int32).max,
                   random_state: int = 0
                   ) -> pd.DataFrame:
    """
    Method to implement random under sampling and random oversampling methods in the <df> dataframe.
    If  n_samples < lower_bound the method will implement a random over sampling generating new data points
    through the <augmenter> transformation.
    If n_samples > upper_bound the method will return a sampled dataframe.

    Args:
        df (pd.DataFrame): Text dataframe, it has to include the column 'text'.
        augmenter (Augmenter): Instance from Augmenter class that performs transformations for data augmentation.
        lower_bound (int): Under sampling ratio.
        upper_bound (int): Over sampling ratio.
        random_state (int): Random seed.

    Returns:
        pd.Dataframe: Sampled according the n_samples and lower_bound configuration.
    """
    n_samples = len(df)

    # Upsample
    if n_samples < lower_bound:
        n_aug = lower_bound - n_samples
        # Add remaining samples to complete the wanted size
        df_tmp = df.sample(n_aug, replace=True, random_state=random_state).copy().reset_index(drop=True)
        # Perform data augmentation over the oversampling samples
        df_tmp['text'] = augmenter.augment(df_tmp['text'].values)
        # Mix the original samples with augmented ones and shuffle
        return pd.concat([df, df_tmp]).sample(frac=1, replace=False, random_state=random_state)

    # Downsample
    if n_samples > upper_bound:
        return df.sample(upper_bound, random_state=random_state)

    return df
