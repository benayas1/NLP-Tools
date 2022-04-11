from typing import Iterable, Union, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from nlp.pt.fitter import TransformersFitter
import scipy
import sklearn
from sklearn.metrics import pairwise_distances
from nlp.pt.dataset import DataCollator, TextDataset
from nlp.pt.model import IntentClassifier
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizerFast
import faiss
import logging as log


class LabelExpander:
    """
    Performs a label propagation calculating the cluster centroids and the distance between them a new unlabeled
    data points. The calculation is based on 3 elements: (1) cluster centroid (2) max ratio and (3) min ratio.
    A new unlabeled data point will be labeled if it is near to any cluster area, between the max and min ratio.
    Otherwise it will be labeled with the unknown id '-1'.
    """
    def __init__(self,
                 threshold: float = 0.99,
                 candidates: int = 1,
                 progress_bar: bool = True):
        """
        Class constructor.

        Args:
            threshold (float): Ratio to determine if any unlabeled data point is near to the centroid.
            candidates (int): Maximum number of candidates to propagate.
            progress_bar (bool): Enable on screen progress bar.
        """
        self.threshold = threshold
        self.candidates = candidates
        self.progress_bar = progress_bar
        self._index_to_label = {}
        self._label_to_index = {}
        # Save results
        self.centroids = None
        self.mins = None
        self.maxs = None

    def fit(self, X: np.ndarray,
            y: Iterable[Union[str, int]]
            ) -> None:
        """
        Calculate the cluster centroids and their corresponding max and min limits, given de data points X and labels y.

        Args:
            X (np.ndarray): Vector of features with size NxD, been N the samples and D the features dimension
            y (Iterable[Union[str, int]]): Iterable with label representations with size NxD.

        """

        # Normalize data
        X = X/np.linalg.norm(X, ord=2, axis=1).reshape(-1, 1)

        shape = (len(np.unique(y)), X.shape[1])

        # Initialize centroids, min and max
        centroids = np.zeros(shape=shape)
        mins = np.zeros(shape=shape)
        maxs = np.zeros(shape=shape)

        #  the sorted elements in the array.
        self._index_to_label = {index: label for index, label in enumerate(set(y))}
        self._label_to_index = {label: index for index, label in self._index_to_label.items()}
        y = np.array([self._label_to_index[y_label] for y_label in y])

        # Calculate centroids, mins and max on each dimension
        for cid in np.unique(y):
            idx = np.argwhere(y == cid).flatten()
            centroids[cid] = X[idx].mean(axis=0)
            mins[cid] = X[idx].min(axis=0)
            maxs[cid] = X[idx].max(axis=0)

        # Save results
        self.centroids = centroids.astype(np.float32)
        self.mins = mins.astype(np.float32)
        self.maxs = maxs.astype(np.float32)

    def transform(self,
                  X: np.ndarray
                  ) -> np.array:
        """
        This method performs an inference using the centroid configuration to assign cluster labels to the X
        features.

        Args:
            X (np.ndarray):  Vector of features with size (NxD), been N the samples and D the features dimension.

        Returns:
            np.array with dimension (N,) with the predicted labels.
        """

        # Normalize data
        X = X/np.linalg.norm(X, ord=2, axis=1).reshape(-1, 1)

        # Calculate distances to centroids
        dist = pairwise_distances(X, self.centroids)

        # Sort distances
        distances = np.argsort(dist)

        # All labels start at -1
        labels = np.full(shape=(len(X),), fill_value=-1)

        # Propagate
        for i in tqdm(range(len(X)), disable=not self.progress_bar):
            v = X[i]
            for j in range(self.candidates):
                # distances[0,j] will contain the nearest "cluster index" for j candidate
                idx = distances[i, j]
                # unless threshold% of the unlabeled data point must be in between the min and max cluster values
                if np.mean(v >= self.mins[idx]) > self.threshold and np.mean(v <= self.maxs[idx]) > self.threshold:
                    labels[i] = idx
                    break

        return np.array([self._index_to_label.get(index_label, -1) for index_label in labels])


class Propagator:
    """
    This class performs a label propagation process following a hybrid approach that combines deep neural networks and
    KNN. Given a set of labeled corpus (gold_corpus, gold_labels) and a set of unlabeled corpus (corpus) it will
    propagate the unlabeled data using the labeled one.

    A transformed model generates embeddings representations that will be used by a KNN model to find the golden
    dataset nearest neighbors in the unlabeled data. A subset of the best candidates in the unlabeled dataset is
    selected to augment the training dataset in the next iteration. The process ends until all the unlabeled dataset is
    processed by the transformer model, providing at the end of the process labels for all the unlabeled dataset.
    """
    def __init__(self,
                 gold_corpus: Iterable[str],
                 gold_labels: Iterable[int],
                 n_classes: int = None,
                 tokenizer: Union[AutoTokenizer, RobertaTokenizerFast] = None,
                 collator: DataCollator = None,
                 device: str = None,
                 progress_bar: bool = True,
                 batch_size: int = 32,
                 alpha: float = 0.99,
                 top_k: int = 20,
                 expansion_policy: str = 'linear',
                 dist_policy: str = 'stratified',
                 max_iter: int = 20,
                 max_length: int = 150,
                 verbose: bool = True):
        """
        Class constructor.

        Args:
            gold_corpus (Iterable[str]): Text corpus of N documents. Those values represent the grand true features.
            gold_labels (Iterable[int]): Label ids of N documents. Those values represent the grand true labels.
            n_classes (int) : Amount of classes to consider.
            tokenizer Union[AutoTokenizer, RobertaTokenizerFast]: tokenizer model.
            collator (DataCollator): Object that parse and process the input data into batches.
            device (str): device type, GPU(cuda) or CPU.
            progress_bar (bool): Flag to show in screen the progress bar
            batch_size (int): Batch size used for training and validation loop.
            alpha (float): Factor applied to the KNN normalized graph that computes the cluster distances.
            top_k: K nighbors to consider in the KNN model.
            expansion_policy (str, optional): Quantity of new utterances added to model training per iteration.
                                              Possible values are:'linear' or 'exponential', depending on whether
                                              we want to maintain a constant absolute value or a constant ratio value
                                              (wrt previous iteration). Defaults to 'linear'.
            dist_policy (str, optional): The way new samples are added to the model training, either by global top
                                         confidence score or by stratified top confidence score. Possible values are
                                         'top' or 'stratified'. Defaults to 'stratified'.
            max_iter: Maximum number of iterations in the gradient iteration to compute the labels weights in the
                propagation process.
            max_length: Maximum length of tokens to consider in tokenization process.
            verbose (book): Enable logs on screen.
        """

        # Checks on labels consistency
        assert len(gold_corpus) == len(gold_labels)
        assert min(gold_labels) == 0
        assert max(gold_labels) + 1 == len(np.unique(gold_labels))
        assert expansion_policy in ['linear', 'exponential']
        assert dist_policy in ['top', 'stratified']

        self.gold_corpus = gold_corpus
        self.gold_labels = gold_labels
        self.n_classes = len(np.unique(self.gold_labels)) if n_classes is None else n_classes

        self.tokenizer = tokenizer
        self.collator = collator
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.progress_bar = progress_bar

        self.batch_size = batch_size
        self.alpha = alpha
        self.top_k = top_k
        self.max_iter = max_iter
        self.max_length = max_length
        self.expansion_policy = expansion_policy
        self.dist_policy = dist_policy
        self.verbose = verbose

        # Create gold data loader
        dataset = TextDataset(self.gold_corpus,
                              self.gold_labels,
                              n_classes=self.n_classes,
                              device=self.device,
                              only_labelled=True)

        self.gold_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       collate_fn=self.collator,
                                                       shuffle=True)
        self.labels = None
        self.weights = None
        self.class_weights = None

    def fit(self,
            corpus: Iterable[str],
            n_times: int,
            fitter: TransformersFitter,
            val_loader: DataLoader = None,
            callbacks: Iterable = None
            ) -> pd.DataFrame:
        """
        Implements the training of a transformer model (fitter.model) n_times to augment the training data set in each
        iteration using sub sets of corpus, until it has been processed completely in the training loop.

        Each iteration involves two training steps for the transform model, one using only the gold dataset, and
        one more using the gold dataset + sub set of corpus. The sub set corpus is selected in each iteration filtering
        the propagated labels with best score.

        Once the model complete the last n_times iteration, will update the instance attributes label, weights
        and class_weights, witch contains the labels ids and weights assigned to the corpus samples.

        Args:
            corpus (Iterable[str]): Text corpus
            n_times (int): Number of iterations to perform in the training loop
            fitter (TransformersFitter): Pytorch fitter developed to handle the training and validation process
                of the transformer model available on fitter.model.
            val_loader (DataLoader): Pytorch loader that contains the validation partition to evaluate the model.
            callbacks (callbacks): Iterable function to track the training information logs (Ex. Weights and Bias)

        Returns:
            pd:dataFrame that contains the training and validation information at epoch level.

        """

        # join corpus
        full_corpus = np.array(list(self.gold_corpus) + list(corpus))

        # Create labels array
        labels = np.full(shape=(len(full_corpus), ), fill_value=-1)
        labels[:len(self.gold_labels)] = self.gold_labels
        gold_labels_idx = np.argwhere(labels != -1).squeeze()
        if self.dist_policy == "stratified":
            unlabelled_idx = [i for i in range(len(labels)) if i not in gold_labels_idx]

        # Init step size and index
        step_size = int((len(corpus) / n_times) + 1)  # fixed step size for the linear policy
        index = len(gold_labels_idx)
        opt_alpha = np.power(len(corpus) / index * 1.0, 1 / n_times) - 1  # factor for exponential policy

        history = []
        p_labels = []
        for i in tqdm(range(n_times), disable=not self.progress_bar):

            self._log(log.INFO, f'Iteration {i + 1}')

            # encode all the data with the current model state
            self._log(log.INFO, 'Extracting features')
            embeddings = self._get_encodings(fitter.model, full_corpus, batch_size=self.batch_size)

            # propagate labels
            self._log(log.INFO, 'Propagating labels')
            labels, weights, class_weights = self._propagate_label(labels,
                                                                   gold_labels_idx,
                                                                   embeddings)
            pct_change = 1-sklearn.metrics.accuracy_score(labels, p_labels) if len(p_labels) > 0 \
                else 1.0-(len(self.gold_labels)/len(labels))
            self._log(log.INFO, f"Pct change in labels: {pct_change}")
            p_labels = labels.copy()

            # Choose the most confident utterances based on label propagation results
            if i < n_times - 1:  # If we are in the last iteration, there's no unlabelled data so no need to go through this part
                if self.expansion_policy == 'linear':
                    index += step_size
                else:
                    index = index*(1+opt_alpha) if i < n_times-1 else len(labels)
                    index = int(index)

                if self.dist_policy == 'top':
                    labels[np.argsort(weights)[::-1][index:]] = -1  # Set the least confident samples back to -1
                    label2quant = {i: v for i, v in zip(range(len(class_weights)),
                                                        np.bincount(labels[weights.argsort()[::-1]][len(gold_labels_idx):index],
                                                                    minlength=self.n_classes) /
                                                        np.sum(np.bincount(labels[weights.argsort()[::-1]][len(gold_labels_idx):index], minlength=self.n_classes),
                                                               keepdims=True))
                                   }
                else:
                    unlabelled_class_weights = np.bincount(labels[unlabelled_idx], minlength=self.n_classes)/np.sum(np.bincount(labels[unlabelled_idx], minlength=self.n_classes), keepdims=True)  # Data available distribution
                    label2goldquant = {i: int(v) for i, v in zip(range(len(class_weights)), np.bincount(labels[gold_labels_idx], minlength=self.n_classes))}
                    label2quant = {i: int(v) for i, v in zip(range(len(class_weights)), (index-len(gold_labels_idx))*unlabelled_class_weights)}
                    null_idxs = np.concatenate([weights.argsort()[::-1][np.argwhere(labels[weights.argsort()[::-1]] == i).reshape(-1)][(label2quant[i]+label2goldquant[i]):] for i in range(len(class_weights))])
                    labels[null_idxs] = -1

            # Calculate some metrics
            self._log(log.INFO, f'Number of labelled examples so far {sum(labels!=-1)}/{len(labels)}.')
            self._log(log.INFO, f'Avg. Confidence {np.mean(weights)}')
            self._log(log.INFO, f'Avg Selected Confidence {np.mean(weights[labels!=-1])}')
            if self.dist_policy == 'top':
                self._log(log.INFO, f'% phrases added per intent without stratified selection:\n{100*np.bincount(labels[weights.argsort()][::-1][len(gold_labels_idx):index])/np.sum(np.bincount(labels[weights.argsort()][::-1][len(gold_labels_idx):index]), keepdims=True)}')
            else:
                self._log(log.INFO, f'% phrases added per intent with stratified selection:\n{100*(np.array(list(label2quant.values()))/np.sum(np.array(list(label2quant.values())), keepdims=True))}')

            history_step = {'iter': i,
                            'ratio_labelled': (labels != -1).mean(),
                            'conf_all': np.mean(weights),
                            'conf_labelled': np.mean(weights[labels != -1]),
                            'pct_change': pct_change}

            # Generate new dataset including the new added samples
            dataset = TextDataset(full_corpus,
                                  labels,
                                  weights=weights,
                                  class_weights=class_weights,
                                  n_classes=self.n_classes,
                                  device=self.device,
                                  only_labelled=True)

            train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=self.batch_size,
                                                       collate_fn=self.collator,
                                                       shuffle=True)

            # Train for 1 more epoch
            fitter.model.train()
            self._log(log.INFO, 'Epoch on already labelled dataset')

            fitter_history = fitter.fit(train_loader,
                                        val_loader=val_loader,
                                        n_epochs=1,
                                        verbose_steps=10)
            history_step['train_all_loss'] = fitter_history['val'].values[-1] if 'val' in fitter_history \
                else fitter_history['train'].values[-1]

            # Another epoch on the gold label
            self._log(log.INFO, 'Epoch on golden dataset')
            # if self.verbose:
            #    print('Epoch on golden dataset')
            fitter_history = fitter.fit(self.gold_loader,
                                        val_loader=val_loader,
                                        n_epochs=1,
                                        verbose_steps=10)
            history_step['train_gold_loss'] = fitter_history['val'].values[-1] if 'val' in fitter_history \
                else fitter_history['train'].values[-1]

            # Append data to history dataframe
            history.append(history_step)

            if callbacks is not None:
                for c in callbacks:
                    c(history_step)

        # Save final results
        self.labels = labels
        self.weights = weights
        self.class_weights = class_weights

        return pd.DataFrame(history)

    def _get_encodings(self,
                       model: IntentClassifier,
                       corpus: Iterable[str],
                       batch_size: int = 128
                       ) -> np.ndarray:
        """
        This method aims to get the encoding vector representation of 'corpus' using the pretrained model 'model'.

        Args:
            model (IntentClassifier): Transformer model (Roberta) which implements a 'encode' method to get compute the
                corpus encodings.
            corpus (Iterable[str]): Text corpus of size (N,) that contains N text documents.
            batch_size (Int): Number of samples to process in each batch iteration when performing encoding.

        Returns:
            np.ndarray with dimensions (N,D) been D the dimension of encoding vectors (768 from Roberta pooler_output)
        """
        # Data
        dataset = TextDataset(data=corpus,
                              device=self.device,
                              only_labelled=False)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=2)

        # Create embeddings
        embeddings = []
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for x in tqdm(train_loader, disable=not self.progress_bar):

                x = self.tokenizer(x,
                                   is_split_into_words=False,
                                   return_offsets_mapping=False,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt',
                                   max_length=self.max_length)

                inputs, masks = x['input_ids'], x['attention_mask']
                inputs = inputs.squeeze().to(self.device).int()
                masks = masks.squeeze().to(self.device).int()

                output = model.encode(inputs, masks).cpu().numpy()
                embeddings.append(output)

        embeddings = np.concatenate(embeddings)

        return embeddings

    def _propagate_label(self,
                         labels: np.array,
                         gold_labels_idx: np.array,
                         features: np.array
                         ) -> Tuple[np.array, np.array, np.array]:
        """
        Propagates labels based on a KNN similarity graph.

        Args:
            labels (np.array): Array of shape (N,) with labels ids. Unlabeled points are represented with -1 value.
            gold_labels_idx (np.array): Array of shape (n,), where n < N. These do not change over the course of the
                propagation, those labels are the grand truth annotations.
            features (np.array): Array of shape (N, n_features) that contains the embedding representation of N samples.

        Returns:
            Tuple(np.array): Returns three arrays p_labels, weights, class_weights: the new labelled array, the
            confidence of each point and the class weights.
        """

        # Create kNN network
        if self.device == 'cpu':
            knn = faiss.IndexFlatIP(features.shape[1])
        else:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = int(torch.cuda.device_count()) - 1
            knn = faiss.GpuIndexFlatIP(res, features.shape[1], flat_config)

        faiss.normalize_L2(features)
        knn.add(features)

        # Run top k for each sample
        gamma = 3  # as per the code from the paper
        D, index_matrix = knn.search(features, self.top_k + 1)
        # both D and I have shape (n_samples, k)
        D = D[:, 1:] ** gamma  # Distance Matrix ignore sample own distance
        index_matrix = index_matrix[:, 1:]  # Index matrix ignore own distance

        # Create the graph, which is a sparse matrix to save memory
        N = features.shape[0]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (self.top_k, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), index_matrix.flatten('F'))),
                                    shape=(N, N))  # the sparse matrix of pairwise similarities vi and vj
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initialize the y vector (called Z) for each class (eq 5 from the paper, normalized with the class size) and
        # apply label propagation
        labelled_idx = np.argwhere(labels != -1).squeeze()  # calculate labelled indices
        Z = np.zeros((N, self.n_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - self.alpha * Wn
        for i in range(self.n_classes):
            cur_idx = labelled_idx[np.where(labels[labelled_idx] == i)]  # np.argwhere(labels[labelled_idx]==i).squeeze()
            y = np.zeros((N,))
            if len(cur_idx) > 0:
                y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=self.max_iter)
            Z[:, i] = f  # compute labels weights according the k-neighbors class-ratio solving ec. Ax=y

        # Antonio's correction on numerical stability issues
        # Handle numerical errors : normalize probs.
        eps = 1e-10
        probs_l1 = np.apply_along_axis(lambda row: (row-(np.min(row)-eps))/(np.max(row)-(np.min(row)-eps)), 1, Z)
        probs_l1 = np.apply_along_axis(lambda row: row/np.sum(row), 1, probs_l1)
        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        entropy = scipy.stats.entropy(probs_l1.T)  # outputs a vector of length N
        weights = 1 - entropy / np.log(self.n_classes)
        weights = weights / np.max(weights)  # normalize in [0,1] # These are sample-weights
        p_labels = np.argmax(probs_l1, axis=1)  # outputs a vector of length N. these are the proposed pseudolabels

        # Keep gold labels
        p_labels[gold_labels_idx] = labels[gold_labels_idx]
        weights[gold_labels_idx] = 1.0  # samples that were already labelled are reset to 1

        # Compute the weight for each class
        class_weights = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            cur_idx = np.where(np.asarray(p_labels) == i)[0]
            if len(cur_idx) > 0:
                class_weights[i] = (float(labels.shape[0]) / self.n_classes) / cur_idx.size

        return p_labels, weights, class_weights

    def _log(self,
             severity: int,
             s: str):
        """
        Enable the logging module.

        Args:
            severity(int): Log level where CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10 and NOTSET=0
            s(str): log message
        """
        if self.verbose:
            print(s)
            log.log(severity, s)
