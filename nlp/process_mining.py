import pandas as pd
from scipy.sparse import csr_matrix
import collections
from typing import List, Union
import numpy as np
from tqdm.auto import tqdm

class ProcessMining():
    def __init__(self, 
                 column_id: str = 'conversationId',
                 column_ts: str = 'ts',
                 column_text: str = 'text',
                 column_intent: str = 'intent',
                 progress_bar: bool = False):

        self.column_id = column_id
        self.column_ts = column_ts
        self.column_text = column_text
        self.column_intent = column_intent
        self.progress_bar = progress_bar
        self.is_fit = False

    def _single_conv(self, 
                     data):
        from_data = data[:-1]
        to_data = data[1:]
        freq = [1] * len(from_data)
        return csr_matrix((freq, (from_data, to_data)), shape=(len(self.intent2idx), len(self.intent2idx)))

    def _get_matrix(self, 
                    data):
        matrix = self._single_conv(data[0])
        for x in data[1:]:
            matrix += self._single_conv(x)
        return matrix

    def get_probabilities(self,
                          sequence: List[str],
                          return_probabilities: bool = True):
        if not self.is_fit:
            raise KeyError('Call fit method first')
        
        if len(sequence) == 0:
            raise ValueError('Empty sequence')

        numeric_sequence = tuple([self.intent2idx[x] for x in sequence])
        
        freq = self.sequence_frequency[numeric_sequence]
        if return_probabilities:
            return freq / sum(self.sequence_frequency.values())
        else:
            return freq

    def get_all_paths(self,
                      return_probabilities: bool = True,
                      threshold: float = 0.0, # 0.1
                      is_above_threshold: bool = True):
        if not self.is_fit:
            raise KeyError('Call fit method first')
        
        paths, freqs = [], []
        for k, v in self.sequence_frequency.items():
            paths.append(list(k))
            freqs.append(v)

        if return_probabilities:
            freqs = (np.array(freqs)/sum(freqs)).tolist()

        if is_above_threshold:
            data_filtered = filter(lambda e: e[1] >= threshold, list(zip(paths, freqs)))
        else:
            data_filtered = filter(lambda e: e[1] < threshold, list(zip(paths, freqs)))

        data_filtered = sorted(data_filtered, key = lambda e: e[1])

        paths = [e[0] for e in data_filtered]
        paths = list(map(lambda x: [self.idx2intent[e] for e in x], paths))
        freqs = [e[1] for e in data_filtered]

        return paths, freqs

    def fit(self, data):
        self.data = data.sort_values(by=[self.column_id, self.column_ts])

        # Calculate unique intents aka nodes
        self.intent2idx = {k: i for i, k in enumerate(data[self.column_intent].unique())}
        self.idx2intent = {i: k for i, k in enumerate(data[self.column_intent].unique())}

        # Get all sequences and calculate its frequency
        all_sequences = [tuple(sequence[self.column_intent].map(self.intent2idx).values) for _, sequence in tqdm(data.groupby(self.column_id), disable=not self.progress_bar)]
        self.sequence_frequency = collections.Counter(all_sequences)

        # Calculate matrix
        self.matrix = self._get_matrix(all_sequences)

        # Flag this as fit
        self.is_fit = True
