from typing import Tuple, Union, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
from sklearn.feature_extraction.text import TfidfVectorizer
import logging as log


class BaseClustering(BaseEstimator, ABC):
    """
    Base class for clustering that performs common methods to merge, impute and parse cluster outputs.
    """
    def __init__(self,
                 metric: str = 'cosine',
                 merge_clusters_thr: float = 0.01,
                 l2_norm: bool = True,
                 remove_negative_silhouette: bool = True,
                 verbose: bool = True,
                 clean_tolerance: float = 0.0):
        """
        Class constructor.

        Args:
            metric (str): Metric used to calculate clusters distance.
            merge_clusters_thr (float): Threshold distance for cluster merging.
            l2_norm (bool): If true the data will be normalized using l2 norm.
            remove_negative_silhouette (bool): If true, clusters with negative silhouette coefficient will be removed.
            verbose (bool): Display on screen method logs.
            clean_tolerance (float): defines how restrictive is the cluster cleaning process. For positive values the
            higher the value more restrictive, for negative lower values less restrictive.

        """
        self.metric = metric
        self.merge_clusters_thr = merge_clusters_thr
        self.l2_norm = l2_norm
        self.remove_negative_silhouette = remove_negative_silhouette
        self.labels_ = []
        self.verbose = verbose
        self.distance_matrix = np.empty(0)
        self.silhouette = np.empty(0)
        self.deviation_metric = np.empty(0)
        self.label_texts_ = []
        self.clean_tolerance = clean_tolerance

    def merge_clusters(self,
                       labels: np.array,
                       features: np.array
                       ) -> np.array:
        """
        Merges clusters with a cosine distance below a certain threshold.
        For each cluster its mean is calculated feature-wise. Then a distance matrix is computed.
        Those cluster pairs with a distance below the threshold will be merged.
        The merging are done in a greedy way.
        After merging all possible pairs, some cluster ID's might be empty, so labels are reclassified
        in order to have all cluster ID's sorted with no gaps.

        Args:
            labels (np.array): Array with shape (n_samples, ) with a cluster ID for each sample.
            features (np.array): Array with shape (n_samples, n_features) with features.

        Returns:
            np.array: A new array with shape (n_samples, ) with the new cluster ID for each sample.
        """

        if self.merge_clusters_thr <= 0.0:
            return labels

        assert len(labels) == len(features)

        while True:
            # Calculate cosine distances for each cluster
            distance_matrix, idx2cid = self._get_distance_matrix(features, labels)

            if len(distance_matrix) == 0:
                break

            # Break the loop if there is nothing else to merge
            if (distance_matrix < self.merge_clusters_thr).sum() <= len(distance_matrix):
                break

            # merge clusters
            already_merged = set()
            for i in range(distance_matrix.shape[0]):
                for j in range(distance_matrix.shape[1]):
                    if j not in already_merged and i != j and distance_matrix[i, j] < self.merge_clusters_thr:
                        self._log(log.DEBUG, f"Merging cluster {idx2cid[j]} into cluster {idx2cid[i]}")
                        labels[labels == idx2cid[j]] = idx2cid[i]
                        already_merged.add(j)

        # reassign cluster_id's
        labels = self._fill_gaps(labels)

        # Save cluster distance matrix for evaluation
        self.distance_matrix, _ = self._get_distance_matrix(features, labels)

        return labels

    # @todo merge: function is staticmethod
    # def _fill_gaps(self, labels: np.ndarray
    @staticmethod
    def _fill_gaps(labels: np.ndarray
                   ) -> np.ndarray:
        """
        Method to re-assign cluster label ids if is needed.
        If the distinct cluster ids in 'labels 'are not consecutive, this method will assign new consecutive ids to the
        labels. Example: if 'labels' is [1,1,1,3,3,3,6,6,6,6,-1] will return [1, 1, 1, 2, 2, 2, 0, 0, 0, 0, -1].

        Args:
            labels (np.ndarray): Array with shape (n_samples, ) with a cluster ids for each sample.

        Returns:
            np.ndarray: Return a vector with same length of 'labels'.
        """

        labels = np.array(labels)
        labels_map = {label: index for index, label in enumerate(np.unique(labels[labels != -1]))}
        return np.array([labels_map.get(label, -1) for label in labels])

    def _get_distance_matrix(self,
                             features: np.ndarray,
                             labels: np.ndarray
                             ) -> Union[Tuple[np.ndarray, dict], Tuple[List, dict]]:
        """
        Estimates the cluster centroids by averaging the clusters values to calculate the clusters
        distances of each centroid against the others using the self.metric.

        Args:
            features (np.ndarray): Array with shape (n_samples, n_features) with features.
            labels (np.ndarray):  Array with shape (n_samples, ) with a cluster ID for each sample.

        Returns:
            Union[Tuple[np.ndarray, dict], Tuple[List, dict]]: returns a tuple, the first component (distance_matrix)
            is a matrix size (n_clusters,n_clusters) with centroid distances. The second component (idx2cid)
            is a dictionary with the clusters labels ids (ignoring the label non cluster-1).
        """
        cluster_embedding = {}
        for cid in np.unique(labels):
            if cid != -1:
                cluster_embedding[cid] = np.mean(features[labels == cid], axis=0)

        # No clusters
        if len(cluster_embedding) == 0:
            return [], {}

        # Generate matrix
        idx2cid = {}
        cluster_matrix = []
        for i, (cid, embedding) in enumerate(cluster_embedding.items()):
            idx2cid[i] = cid
            cluster_matrix.append(embedding)
        cluster_matrix = np.stack(cluster_matrix)

        # Calculate cosine distances
        distance_matrix = pairwise_distances(cluster_matrix, metric=self.metric)
        return distance_matrix, idx2cid

    def fit_predict(self,
                    X: np.ndarray
                    ) -> np.ndarray:
        """
        Implements training and inference methods building clustering models and performing inference of this model
        in the X features.

        Args:
            X (np.ndarray): Array with shape (n_samples, n_features) with features.

        Returns:
            np.ndarray: return a vector with the labels ids of each found cluster in the documents X.
        """
        if not (type(X) is np.ndarray):
            raise Exception("Please pass a numpy array")

        if self.l2_norm:
            X = X/np.linalg.norm(X, ord=2, axis=1).reshape(-1, 1)

        self._fit(X)

        self.labels_ = self.merge_clusters(self.labels_, X)

        if self.remove_negative_silhouette and len(np.unique(self.labels_)) > 1:
            # Calculate silhouette
            silhouette = silhouette_samples(X, labels=self.labels_)

            # Calculate deviation metric
            deviation_metric = np.zeros(shape=(len(self.labels_, )))
            for cid in np.unique(self.labels_):
                if cid == -1:
                    continue
                idx = np.argwhere(self.labels_ == cid).flatten()
                # Computes the embedding deviation against their own cluster (a deviation for each n_sample)
                matrix = pairwise_distances(X[idx])
                deviation_metric[idx] = (matrix.mean(axis=0) - matrix.mean() - matrix.var())*-1

            # Clusters with negative silhouette are reset to -1
            for cid in np.unique(self.labels_):
                idx = np.argwhere(self.labels_ == cid).flatten()
                if silhouette[idx].mean() < 0:
                    self.labels_[idx] = -1

            # Clean some samples with bad scoring
            self.labels_[(silhouette <= self.clean_tolerance) * (deviation_metric <= self.clean_tolerance)] = -1

            # Remove gaps
            self.labels_ = self._fill_gaps(self.labels_)
            self.silhouette = silhouette
            self.deviation_metric = deviation_metric

        return self.labels_

    def get_topics(self,
                   X: List[str],
                   threshold: int = 0.25,
                   **kwargs
                   ) -> np.ndarray:
        """
        Assign names for clusters ids using the words in the topic with tfidf ratio higher than threshold.
        This method assumes the fit method was already implemented.

        Args:
            X (List[str]): List of documents to process, where each document is related with a cluster in
                self.labels_.
            threshold: All the words with tfidf score over this tfidf threshold will be considered in the clustering
                naming.
            **kwargs: Dictionary with method attributes for scikit learn tfidf transformation "TfidfVectorizer".

        Returns:
            np.ndarray: array with string names with shape (n_samples, ). Where each name corresponds to a unique
                cluster id in self.labels_.
        """
        if len(self.labels_) == 0:
            raise Exception('Call fit method first')

        assert len(self.labels_) == len(X)

        if len(np.unique(self.labels_)) <= 1:
            return ['Unlabelled'] * len(X)

        # Create a dataframe with cluster labels and texts
        df = pd.DataFrame({'label': self.labels_, 'text': X})

        # Generate a single text for each cluster
        cluster_texts = df[df['label'] != -1].groupby('label').apply(lambda g: ' '.join(g['text']))

        # Create TF-IDF
        vectorizer = TfidfVectorizer(**kwargs)
        vectors = vectorizer.fit_transform(cluster_texts).todense().tolist()
        df_tf_idf = pd.DataFrame(vectors, columns=vectorizer.get_feature_names())
        df_tf_idf['summary'] = df_tf_idf.apply(lambda x: ' '.join(x[x > threshold].index), axis=1)

        # Save text for each cluster_id
        self.label_texts_ = df_tf_idf['summary'].to_dict()
        self.label_texts_[-1] = 'Unlabelled'

        # Apply the mapping to every individual text
        df['label_text'] = df['label'].map(self.label_texts_)

        return df['label_text'].values

    @abstractmethod
    def _fit(self,
             X: np.ndarray):
        """
        Performs a training for clustering models using the X features.

        Args:
            X(np.ndarray): Array with shape (n_samples, n_features) with features.

        Returns:
                self: return the class instance.
        """
        raise NotImplementedError('Implement method _fit')

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
            log.log(severity, s)


class IDensity(BaseClustering):
    """
    Iteratively adapt dbscan or optics parameters for unbalanced data (text) clustering
    The change of core parameters of DBSCAN i.e. distance and minimum samples parameters are changed smoothly to
    find high to low density clusters. At each iteration distance parameter is increased by 0.01 and minimum samples
    are decreased by 1. The algorithm uses cosine distance for cluster creation
    """
    def __init__(self,
                 initial_max_distance: float = 0.10,
                 initial_minimum_samples: int = 20,
                 delta_distance: float = 0.01,
                 delta_minimum_samples: int = 1,
                 max_iteration: int = 5,
                 threshold: int = 1000,
                 algorithm: str = 'optics',
                 metric: str = 'euclidean',
                 l2_norm: bool = True,
                 progress_bar: bool = False,
                 merge_clusters_thr: float = 0.01,
                 remove_negative_silhouette: bool = True,
                 logging_level: int = None,
                 n_jobs: int = 1):
        """
        Class constructor.

        Args:
            initial_max_distance (float): The maximum distance between two samples for one to be considered as in the
                neighborhood of the other.
            initial_minimum_samples (int): The number of samples in a neighborhood for a point to be considered as a
                core point. This includes the point itself.
            delta_distance (float): Distance to increase initial_max_distance in each iteration step.
            delta_minimum_samples (int): Number of samples to decrease initial_minimum_samples in each iteration step.
            max_iteration (int): Maxim number of iterations to do.
            threshold (int): Cluster frequency threshold. Clusters with high frequency will be assigned the non
                cluster id -1
            algorithm (str): Clustering algorithm to implement 'optics' or 'dbscan'.
            metric (str): Metric to compute distances and similarities between clusters and samples.
            l2_norm (Bool): If True enables l2 feature normalization.
            progress_bar (bool): If True enable progress on screen progress bar
            merge_clusters_thr (float): Similar clusters with distance below the threshold will be merged
            remove_negative_silhouette: If True labels with negative silhouette coefficient (and negative
                deviation_metric) will be labeled as non cluster (-1).
            logging_level (int): ID to identify the log level priority.
            n_jobs (int): The number of parallel jobs to run for neighbors search. None means 1 unless -1 means all.
        """

        super().__init__(metric, merge_clusters_thr, l2_norm, remove_negative_silhouette, logging_level)
        self.initial_max_distance = initial_max_distance
        self.initial_minimum_samples = initial_minimum_samples
        self.delta_distance = delta_distance
        self.delta_minimum_samples = delta_minimum_samples
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.algorithm = algorithm
        self.progress_bar = progress_bar
        self.labels_ = None
        self.n_jobs = n_jobs

    def _fit(self,
             X: np.ndarray):
        """
        Performs an iterative training/inference process building in each step different cluster models. Each iteration
        builds a cluster model using the remaining data without label cluster assignation. In each iteration the
         method overwrites the found clusters ids in the variable self.labels_.

        Args:
            X(np.ndarray): Array with shape (n_samples, n_features) with features.
        """
        self.labels_ = np.full((len(X),), -1)
        next_cluster_id = 0

        for i in tqdm(range(self.max_iteration), disable=not self.progress_bar):

            self._log(log.INFO, f"Iteration {i}. Max EPS = {self.initial_max_distance} | Min Samples = "
                                f"{self.initial_minimum_samples} ")

            assert len(X) == len(self.labels_)

            # work only with the unlabelled datapoints
            features = X[self.labels_ == -1]
            self._log(log.INFO, f"Samples in this iteration: {len(features)}/{len(X)}")

            if len(features) < self.initial_minimum_samples:
                self._log(log.INFO,
                          f"Total samples {len(features)} less than minimum samples {self.initial_minimum_samples}")
                break

            if self.algorithm.lower() == 'optics':
                clustering = OPTICS(max_eps=self.initial_max_distance,
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)

            if self.algorithm.lower() == 'dbscan':
                clustering = DBSCAN(eps=self.initial_max_distance,
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)
            #  todo Merge: unused clustering technique
            """
            if self.algorithm.lower() == 'hdbscan':
                clustering = hdbscan.HDBSCAN(cluster_selection_epsilon=self.initial_max_distance,
                                             min_cluster_size=self.initial_minimum_samples,
                                             metric='precomputed' if self.metric == 'cosine' else self.metric)
            """

            if not clustering or clustering is None:
                raise Exception('A valid clustering algorithm must be passed')

            # Calculate labels
            labels = clustering.fit_predict(features)
            self._log(log.DEBUG, f"Clusters found in this iteration: {len(np.unique(labels[labels != -1]))}")

            # Adjust labels to keep unlabelled clusters with cardinality above the threshold
            label_freq = Counter(labels)
            labels = np.array([-1 if label_freq[label] > self.threshold or label == -1 else label + next_cluster_id for
                               label in labels])
            self._log(log.DEBUG, f"Clusters after threshold correction: {len(np.unique(labels[labels != -1]))}")

            # Update new labels from this iteration
            assert len(self.labels_[self.labels_ == -1]) == len(labels)
            self.labels_[self.labels_ == -1] = labels
            self._log(log.INFO,
                      f"Total clusters after this iteration: {len(np.unique(self.labels_[self.labels_ != -1]))}")

            # Calculate the next upcoming cluster id
            next_cluster_id = np.max(self.labels_) + 1

            # Update hyperparameters values
            self.initial_max_distance += self.delta_distance
            self.initial_minimum_samples -= self.delta_minimum_samples

            if self.initial_minimum_samples <= 2:
                self._log(log.INFO, "Minimum samples is less than 2")
                break

        return self


class StepsDensity(BaseClustering):
    """
    Processes density clustering iteratively in chunks to overcome memory limitations.
    In every iteration it samples a subset from the corpus and clusters it.
    Labelled examples will not be picked up in further iterations.
    Unlabelled examples will return to the pool of unlabelled examples.
    The loop finishes when all samples were labelled or after a number of iterations.
    Can handle unlimited size corpus.
    """

    def __init__(self,
                 initial_max_distance: float = 0.10,
                 initial_minimum_samples: int = 20,
                 delta_distance: float = 0.01,
                 delta_minimum_samples: int = 1,
                 max_iteration: int = 5,
                 threshold: int = 1000,
                 max_samples: int = 20000,
                 algorithm: str = 'optics',
                 metric: str = 'euclidean',
                 l2_norm: bool = True,
                 progress_bar: bool = False,
                 merge_clusters_thr: float = 0.01,
                 logging_level: int = None,
                 # todo Merge: This parameters help to set up the test uses cases, and also is susceptible for tuning
                 #  specially when the model returns the same df_samples index in many iterations.
                 samples_early_stop: int = 5,
                 n_jobs: int = 1):
        """
        Class constructor

        Args:
            initial_max_distance (float): The maximum distance between two samples for one to be considered as in the
                neighborhood of the other.
            initial_minimum_samples (int): The number of samples in a neighborhood for a point to be considered as a
                core point. This includes the point itself.
            delta_distance (float): Distance to increase initial_max_distance in each iteration step.
            delta_minimum_samples (int): Number of samples to decrease initial_minimum_samples in each iteration step.
            max_iteration (int): Maxim number of iterations to do.
            threshold (int): Cluster frequency threshold. Clusters with high frequency will be assigned the non
                cluster id -1.
            max_samples:
            algorithm (str): Clustering algorithm to implement 'optics' or 'dbscan'.
            metric (str): Metric to compute distances and similarities between clusters and samples.
            l2_norm (Bool): If True enables l2 feature normalization.
            progress_bar (bool): If True enable progress on screen progress bar
            merge_clusters_thr (float): Similar clusters with distance below the threshold will be merged
            logging_level (int): ID to identify the log level priority.
            samples_early_stop (int): Minimum amount of non labeled data points to process. The fit loop will stop early
                if the remaining non labeled data points is under this value.
            n_jobs (int): The number of parallel jobs to run for neighbors search. None means 1 unless -1 means all.
        """

        super().__init__(metric, merge_clusters_thr, l2_norm, logging_level)

        self.initial_max_distance = initial_max_distance
        self.initial_minimum_samples = initial_minimum_samples
        self.delta_distance = delta_distance
        self.delta_minimum_samples = delta_minimum_samples
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.max_samples = max_samples
        self.algorithm = algorithm
        self.progress_bar = progress_bar
        self.labels_ = None
        self.samples_early_stop = samples_early_stop
        self.n_jobs = n_jobs

    def _fit(self,
             X: Union[np.ndarray, list]
             ):
        """
        This method implements an iterative training/inference methods of clustering models labelling the X features.
        The labels will be assigned in iterations taking random samples of X, labeling iteratively the remaining
        non labeled samples.

        Args:
            X (Union[np.ndarray, list]):  Array with shape (n_samples, n_features) with features.
        """

        if not (type(X) is np.ndarray or type(X) is list):
            raise Exception("Please pass a list of string or a list of feature vectors.")

        sample_size = min(self.max_samples, len(X))

        # samples: df_labels will accumulate the results of each iteration,  df_sample the iteration's temporal results
        df_labels = pd.DataFrame(data={'label': np.full((len(X),), -1)})
        df_sample = df_labels.sample(sample_size)

        # init control parameters
        next_cluster_id = 0
        reducing_step = False

        for i in tqdm(range(self.max_iteration), disable=not self.progress_bar):

            assert len(X) == len(df_labels)

            # work only with the unlabelled datapoints
            features = X[df_sample.index]

            print(f'Clustering on {len(df_sample)} samples')

            # stop if number of samples is really small
            if len(features) < self.samples_early_stop:
                break

            if self.algorithm.lower() == 'optics':
                clustering = OPTICS(max_eps=self.initial_max_distance,
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)
            if self.algorithm.lower() == 'dbscan':
                clustering = DBSCAN(eps=self.initial_max_distance,
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)
            #  todo Merge: unused clustering technique
            """
            if self.algorithm.lower() == 'hdbscan':
                clustering = hdbscan.HDBSCAN(cluster_selection_epsilon=self.initial_max_distance,
                                             min_cluster_size=self.initial_minimum_samples,
                                             metric='precomputed' if self.metric == 'cosine' else self.metric)
            """
            if not clustering or clustering is None:
                raise Exception('A valid clustering algorithm must be passed')

            # Calculate labels
            labels = clustering.fit_predict(features)

            # Adjust labels to keep unlabelled clusters with cardinality above the threshold
            label_freq = Counter(labels)
            labels = [-1 if label_freq[label] > self.threshold or label == -1 else label + next_cluster_id
                      for label in labels]

            # update the sample labels with new labels
            df_sample['label'] = labels
            self._log(log.INFO,
                      f"Number of samples labelled in this iteration { len(df_sample[df_sample['label'] != -1]) } ")
            # Accumulate results saving the found clusters on df_labels
            df_labels.iloc[df_sample.index, 0] = df_sample['label']
            # filter the remaining samples without assigned cluster (-1)
            df_sample = df_sample[df_sample['label'] == -1]
            df_sample = df_sample.sample(int(len(df_sample)/2))
            # get the unlabeled data on df_labels that is not in df_sample
            df_available = df_labels.iloc[df_labels.index.difference(df_sample.index)]
            df_available = df_available[df_available['label'] == -1]

            # keep only current unassigned samples and pick new samples to reach sample_size
            # If not enough unlabelled samples available, pick them all
            if len(df_available) <= (sample_size - len(df_sample)):
                reducing_step = True
                df_sample = df_labels[df_labels['label'] == -1]
            else:
                df_sample = pd.concat([df_sample, df_available.sample(sample_size-len(df_sample))])

            # Calculate the next upcoming cluster id
            next_cluster_id = np.max(df_labels['label']) + 1

            self._log(log.INFO, f"Total labelled samples {len(df_labels[df_labels['label']!=-1])}")

            # update cluster parameters only when reducing step
            if reducing_step:
                self.initial_max_distance += self.delta_distance
                self.initial_minimum_samples -= self.delta_minimum_samples

            if self.initial_minimum_samples == 2:
                break

        self.labels_ = df_labels['label'].values
        # todo Merge: Depending on the samples in each iteration data points from the same cluster could be named with
        #   with different cluster id, is needed to join those similar clusters. Example from test use case:
        #   array([2, 2, 2, (3), 1, 1, 1, (4), 5, 0, 0, 5] instead array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
        #   The challenge are the clusters with different density will be joined, to solve this we can perform this step
        #   on the above using self.merge_clusters_thr = self.initial_max_distance (decreasing this value each step).
        #   Another alternative is to perform inference in the whole X in each iteration in the loop
        self.labels_ = self.merge_clusters(self.labels_, X)
        return self
