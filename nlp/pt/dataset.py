import torch
import numpy as np
from typing import List, Tuple


class TextDataset(torch.utils.data.Dataset):
    """
    Dataset Class providing data to the model
    """
    def __init__(self,
                 data,
                 labels=None,
                 weights=None,
                 class_weights=None,
                 n_classes=None,
                 device='cuda',
                 only_labelled=True):
        super().__init__()

        if not isinstance(data, np.ndarray):
            data = data

        self.data = data
        self.labels = labels

        self.device = device
        self.only_labelled = only_labelled

        # Number of classes
        if (len(np.array(labels).shape) > 1) & (n_classes is not None):
            assert n_classes == labels.shape[-1], "Number of classes and shape of binarised labels must be coherent."
            classes = list(range(n_classes))
        elif (len(np.array(labels).shape) > 1) & (n_classes is None):
            classes = list(range(labels.shape[-1]))
        else:
            classes = list(range(n_classes)) if n_classes is not None else np.unique(self.labels)

        # Class weights
        if len(np.array(labels).shape) == 1:
            if class_weights is None:
                self.class_weights = np.ones(shape=(len(classes),))
            else:
                if isinstance(class_weights, str):
                    if class_weights == 'auto' or class_weights == 'balanced':
                        labels_onehot = np.zeros((len(labels), len(classes)))
                        labels_onehot[np.arange(len(labels)), labels] = 1
                        self.class_weights = np.sum(labels_onehot, axis=0)/np.sum(labels_onehot)
                        self.class_weights = {k: v for k, v in zip(classes, self.class_weights)}
                    else:
                        raise ValueError(f"Not valid class weights string {class_weights}. Must be either 'auto' or 'balanced'.")
                else:
                    self.class_weights = class_weights
                    assert len(self.class_weights) == len(classes), f'Class weights vector length ({len(self.class_weights)}) must be equal to the number of classes ({len(classes)})'
        else:
            if class_weights is None:
                self.class_weights = np.ones(shape=(len(classes),))
            else:
                if isinstance(class_weights, str):
                    if class_weights == 'auto' or class_weights == 'balanced':
                        self.class_weights = np.sum(self.labels, axis=0)/np.sum(self.labels)
                        self.class_weights = {k: v for k, v in zip(classes, self.class_weights)}
                    else:
                        raise ValueError(f"Not valid class weights string {class_weights}. Must be either 'auto' or 'balanced'.")
                else:
                    self.class_weights = class_weights
                    assert len(self.class_weights) == len(classes), f'Class weights vector length ({len(self.class_weights)}) must be equal to the number of classes ({len(classes)})'

        # If we want to avoid serving samples with label == -1 (unlabelled)
        if self.only_labelled:
            if labels is None:
                raise Exception('Labels must be provided')

            # Labels
            if not isinstance(self.labels, np.ndarray):
                self.labels = np.array(self.labels)

            self.weights = np.ones(len(labels)) if weights is None else weights

            if len(np.array(labels).shape) == 1:
                labelled_indices = np.argwhere(labels != -1).squeeze()  # avoid label -1
                self.labels = self.labels[labelled_indices]
                self.data = self.data[labelled_indices]

                # Weights
                if not isinstance(self.weights, np.ndarray):
                    self.weights = np.array(self.weights)
                self.weights = self.weights[labelled_indices]
            else:
                labelled_indices = np.sum(labels, axis=-1) > 0
                self.labels = self.labels[labelled_indices, :]
                self.data = self.data[labelled_indices]

                # Weights
                if not isinstance(self.weights, np.ndarray):
                    self.weights = np.array(self.weights)
                self.weights = self.weights[labelled_indices]

        print(f'Number of samples: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        # Text
        t = self.data[key]

        # This is the case when we just want to encode data
        if self.only_labelled == False:
            return t

        # labels
        label = self.labels[key]

        # weight
        w = self.weights[key]
        if len(np.array(self.labels).shape)>1:
            w = w * np.mean(self.class_weights[label==1])
        else:
            w = w * self.class_weights[label]

        return t, label, w


class DataCollator():
    """
    Data Collator object that batches NLP data.
    It tokenizes texts using HuggingFace Transformers. It can also generate NER labels using a spacy pipeline
    """
    def __init__(self,
                 tokenizer,
                 nlp=None,
                 augmenter=None,
                 tag2id=None,
                 ner: bool = False,
                 max_length: int = 100):
        """
        Args:
            tokenizer ([type]): HuggingFace Tokenizer object
            nlp ([type], optional): Spacy pipeline. Required for NER labels. Defaults to None.
            tag2id (dict, optional): Mapping for NER labels. Defaults to None.
            ner (bool, optional): Bool value indicating whether to produce NER labels or not. Defaults to False.
            max_length (int, optional): Defaulted max length of tokenized sentences. Defaults to 100.
        """

        self.tokenizer = tokenizer
        self.nlp = nlp
        self.augmenter = augmenter
        self.tag2id = tag2id
        self.ner = ner
        self.max_length = max_length

    def encode_tags_char(self,
                         labels: List[Tuple[str, int, int]],
                         offsets: List[Tuple[int, int]]) -> List[List[int]]:
        """
        This function generates NER labels by aligning Spacy tokens to HuggingFace Tokens.

        Args:
            labels (List[Tuple[str,int,int]]): List of labels generated by spacy
            offsets (List[Tuple[int,int]]): Offset mapping generated by HuggingFace tokenizer

        Returns:
            List[List[int]]: A list of list of NER labels. Each list represents the labels for each text
        """

        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, offsets):
            # create an empty array of 0
            doc_enc_labels = np.zeros(len(doc_offset), dtype=int)
            arr_offset = np.array(doc_offset)

            for label, start, end in doc_labels:
                selected_tokens = (arr_offset[:, 0] >= start) & (arr_offset[:, 1] <= end) & (arr_offset[:, 1] != 0)
                insert_labels = [self.tag2id['B'+label]]+[self.tag2id['I'+label]]*(sum(selected_tokens)-1)
                doc_enc_labels[selected_tokens] = insert_labels

            # set -100 label for non relevant tokens
            selected_tokens = (arr_offset[:, 1] == 0)
            doc_enc_labels[selected_tokens] = -100

            # set labels whose first offset position is 0 and the second is not 0
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def __call__(self, data):
        """
        Generates a batch of data

        Args:
            data (List of tuples): List of Tuples containing x, y, and weights for each sample in the batch

        Returns:
            dict: A dictionary containing a batch of data. Features goes on 'x' member, labels goes on 'y' and weights on 'w'
        """
        texts, y, w = list(zip(*data))

        # Augment data if augmenter provided
        if self.augmenter is not None:
            texts = self.augmenter.augment(texts)

        # create tokens in numpy format first
        tokens = self.tokenizer(list(texts),
                                is_split_into_words=False,
                                return_offsets_mapping=True,
                                padding=True,
                                truncation=True,
                                return_tensors='np',
                                max_length=self.max_length)

        # intent classifier label
        y_intent = torch.tensor(y)

        # weights
        w = torch.tensor(w)

        # prepare tokens
        x = {'input_ids': torch.tensor(tokens['input_ids']),
             'attention_mask': torch.tensor(tokens['attention_mask'])}

        # calculate entity labels and positions
        if self.ner:
            y_ner = [[(ent.label_, ent.start_char, ent.end_char) for ent in self.nlp(doc).ents] for doc in texts]
            y_ner = torch.tensor(self.encode_tags_char(y_ner, tokens.offset_mapping))
            return {'x': x, 'y': [y_intent, y_ner], 'w': w}

        return {'x': x, 'y': y_intent, 'w': w}
