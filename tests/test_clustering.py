import pandas as pd
import numpy as np
import pytest
from nlp.clustering import StepsDensity, IDensity
import random


@pytest.fixture(scope='module')
def intent_dataset():
    # Data centers centers = [(0.20, 0.70, 0.10),  (0.50, 0.20, 0.40), (0.80, 0.70, 0.60)]
    df_intent = pd.DataFrame({
        'text': ["Hello I want to ask something",
                 "Hi I need your help please",
                 "Good morning I have a question",
                 "Hi good night can you help me",
                 "Thank you very much",
                 "Perfect thank you for your help",
                 "Please accept my best thanks",
                 "I really appreciate it",
                 "I have a issue",
                 "I have problem to solve",
                 "Looks like an error",
                 "The problem is still there"],
        'topics': ['good help hi', 'good help hi', 'good help hi', 'good help hi',
                   'thank you', 'thank you', 'thank you', 'thank you',
                   'have problem', 'have problem', 'have problem', 'have problem'],
        'labels': [0, 0, 0, 0,
                   1, 1, 1, 1,
                   2, 2, 2, 2],
        'embeddings': [[0.42078586,  0.58842285,  0.17590588],
                       [0.02899552,  0.42661827,  0.13829967],
                       [0.20962776,  0.64575975,  0.22078677],
                       [0.00421063,  0.70901880,  0.36140828],
                       [0.45147469,  0.03397876,  0.49858255],
                       [0.55038903,  0.00470465,  0.66285584],
                       [0.22632572,  0.02056206,  0.43908140],
                       [0.40848973,  0.00506694,  0.25629213],
                       [0.97404063,  1.00000000,  0.64185808],
                       [0.75987326,  0.80688113,  0.53990404],
                       [0.73426645,  0.53367722,  0.51027641],
                       [0.78692702,  0.88863130,  0.62596921]
                       ]})
    return df_intent


@pytest.fixture
def intent_dataset_with_outliers(intent_dataset):
    df_intent_outliers = pd.DataFrame({
        'text': ["This is the first outlier example",
                 "This is the second outlier example"],
        'topics': ["Unlabelled",
                   "Unlabelled"],
        'labels': [-1,
                   -1],
        'embeddings': [[0.50000000, 0.95000000, 0.95000000],
                       [0.90000000, 0.10000000, 0.10000000]]})
    return pd.concat([intent_dataset, df_intent_outliers])


@pytest.fixture
def intent_dataset_with_noise(intent_dataset):
    df_intent_noise = pd.DataFrame({
        'text': ["This is the first noisy point"],
        'labels': [-1],
        'topics': ["Unlabelled"],
        'embeddings': [[0.28753736, 0.30426651, 0.34415156]]})
    return pd.concat([intent_dataset, df_intent_noise])


def get_test_data(df):
    return np.vstack(np.array(df['embeddings'])), np.array(df['labels']), np.array(df['topics']), \
           df['text'].values


class ClusterDensityCommonAsserts:

    model_attributes = ['metric', 'merge_clusters_thr', 'l2_norm', 'remove_negative_silhouette', 'verbose']

    @staticmethod
    def parse_cluster_ids(clusters_ids):
        clusters_ids_map = {-1: -1}
        id_counter = 0
        for clusters_id in clusters_ids:
            if clusters_id not in clusters_ids_map:
                clusters_ids_map[clusters_id] = id_counter
                id_counter += 1
        clusters_id_mapped = [clusters_ids_map[clusters_id] for clusters_id in clusters_ids]
        return clusters_id_mapped

    def check_model_attributes(self, cluster_instance):
        for model_attribute in self.model_attributes:
            assert hasattr(cluster_instance, model_attribute)

    @staticmethod
    def check_topics_labeling(predicted_topics, expected_topics):
        # Test predicted length
        assert len(expected_topics) == len(predicted_topics)
        # Test all the cluster was named
        assert all([isinstance(text_label, str) for text_label in predicted_topics])
        # Test the cluster names correspond to expected values
        assert all(predicted_topics == expected_topics)

    def check_predicted_data(self, predicted_clusters_ids, expected_cluster_ids):
        # Check data type
        assert isinstance(predicted_clusters_ids, np.ndarray)
        # Check amount clusters
        assert len(set(predicted_clusters_ids)) == len(set(expected_cluster_ids))
        # Check bucket_id are named without gaps
        assert sorted(set(predicted_clusters_ids)) == list(range(predicted_clusters_ids.min(),
                                                                 predicted_clusters_ids.max()+1))
        # Check all labels corresponds to expected values
        clusters_ids_mapped = self.parse_cluster_ids(predicted_clusters_ids)
        assert len(clusters_ids_mapped == expected_cluster_ids)


@pytest.mark.usefixtures("random_config")
class TestStepDensity(ClusterDensityCommonAsserts):

    def test_clean_clusters(self, intent_dataset):
        # Test find all clusters when data has not outlier
        i_density = IDensity(
            initial_max_distance=0.3,  # dbscan cluster_selection_epsilon
            initial_minimum_samples=2,  # dbscan min_cluster_size
            delta_distance=0.01,
            delta_minimum_samples=2,
            max_iteration=5,
            threshold=1000,
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            remove_negative_silhouette=False,
            logging_level=None)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(df=intent_dataset)
        clusters_ids = i_density.fit_predict(embeddings_features)
        cluster_topics = i_density.get_topics(texts, threshold=0.25)

        # Perform asserts
        self.check_model_attributes(cluster_instance=i_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)

    def test_cluster_outliers(self, intent_dataset_with_outliers):
        # Test find all clusters when data has outlier
        i_density = IDensity(
            initial_max_distance=0.3,  # dbscan cluster_selection_epsilon
            initial_minimum_samples=2,  # dbscan min_cluster_size
            delta_distance=0.01,
            delta_minimum_samples=2,
            max_iteration=5,
            threshold=1000,
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            remove_negative_silhouette=False,
            logging_level=None)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(
            df=intent_dataset_with_outliers)
        clusters_ids = i_density.fit_predict(embeddings_features)
        cluster_topics = i_density.get_topics(texts, threshold=0.25)

        # Perform asserts
        self.check_model_attributes(cluster_instance=i_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)

    def test_cluster_merge(self, intent_dataset):
        # Test merge method works reducing the amount cluster when merge_clusters_thr increase
        i_density = IDensity(
            initial_max_distance=0.45,  # dbscan cluster_selection_epsilon
            initial_minimum_samples=2,  # dbscan min_cluster_size
            delta_distance=0.01,
            delta_minimum_samples=2,
            max_iteration=5,
            threshold=1000,
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.7,
            remove_negative_silhouette=False,
            logging_level=None)
        embeddings_features, _, _, _ = get_test_data(df=intent_dataset)
        clusters_ids = i_density.fit_predict(embeddings_features)
        expected_labels = 2
        predicted_labels = set(clusters_ids)
        # Test cluster was already merged
        assert len(predicted_labels) == expected_labels
        # Test clas fill the gaps using consecutive ids
        assert len(set(range(0, len(predicted_labels))).intersection(predicted_labels))
        # Perform default asserts
        self.check_model_attributes(cluster_instance=i_density)

    def test_filter_negative_silhouette(self, intent_dataset_with_noise):
        i_density = IDensity(
            initial_max_distance=0.3,
            initial_minimum_samples=1,  # to force each point unless one cluster (avoid return -1 cluster)
            delta_distance=0.01,
            delta_minimum_samples=2,
            max_iteration=5,
            threshold=1000,
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            remove_negative_silhouette=True,
            logging_level=True)
        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(intent_dataset_with_noise)
        clusters_ids = i_density.fit_predict(embeddings_features)
        cluster_topics = i_density.get_topics(texts, threshold=0.25)
        # Test noisy point is converted in unknown cluster(-1) cluster
        assert clusters_ids[-1] == expected_cluster_ids[-1]
        # Perform asserts
        self.check_model_attributes(cluster_instance=i_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)


@pytest.mark.usefixtures("random_config")
class TestIDensity(ClusterDensityCommonAsserts):

    def test_clean_clusters_rigid(self, intent_dataset):
        # Test forcing _fit method to create clusters only if sampled data contains at least cluster 4 data points (all)
        s_density = StepsDensity(
            initial_max_distance=0.3,
            initial_minimum_samples=4,
            # To force in each iteration to build a cluster if all the data points was sampled
            delta_distance=0.01,
            delta_minimum_samples=1,
            max_iteration=5,
            threshold=1000,
            max_samples=8,  # 8 To force the fit routine to iterate more than once
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            logging_level=None)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(df=intent_dataset)
        clusters_ids = s_density.fit_predict(embeddings_features)
        cluster_topics = s_density.get_topics(texts, threshold=0.25)

        # Perform asserts
        self.check_model_attributes(cluster_instance=s_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)

    def test_clean_clusters_flexible(self, intent_dataset):
        # Test forcing the model to get more labels that expected
        s_density = StepsDensity(
            initial_max_distance=0.3,
            initial_minimum_samples=2,  # To force create more than 3 groups when only two pints ara available
            delta_distance=0.01,
            delta_minimum_samples=1,
            max_iteration=5,
            threshold=1000,
            max_samples=8,  # To force create more than 3 groups at the end of the fit loop
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.35,  # To force joining similar clusters in the final iteration
            logging_level=None,
            samples_early_stop=3)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(df=intent_dataset)
        clusters_ids = s_density.fit_predict(embeddings_features)
        cluster_topics = s_density.get_topics(texts, threshold=0.25)

        # Perform asserts
        self.check_model_attributes(cluster_instance=s_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)

    def test_cluster_outliers(self, intent_dataset_with_outliers):
        # Test using anomalies to reject them
        s_density = StepsDensity(
            initial_max_distance=0.3,
            initial_minimum_samples=4,  # With 4 will find the first 2 groups, with 3 the last one.
            delta_distance=0.01,
            delta_minimum_samples=1,
            max_iteration=5,
            threshold=1000,
            max_samples=14,  # 8 To force the fit routine in just one iteration
            algorithm='dbscan',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            logging_level=None)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(
            df=intent_dataset_with_outliers)
        clusters_ids = s_density.fit_predict(embeddings_features)
        cluster_topics = s_density.get_topics(texts, threshold=0.25)

        # Perform default asserts
        self.check_model_attributes(cluster_instance=s_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)
        # Test outliers was not labeled
        assert all(np.where(clusters_ids == -1)[0] == np.where(expected_cluster_ids == -1)[0])

    def test_clean_optics(self, intent_dataset):
        # Test Optics method
        s_density = StepsDensity(
            initial_max_distance=0.4,
            initial_minimum_samples=2,
            delta_distance=0.01,
            delta_minimum_samples=1,
            max_iteration=3,  # To force the fit routine max two iteration to avoid labeling the outliers
            threshold=1000,
            max_samples=14,
            algorithm='optics',
            metric='euclidean',
            l2_norm=False,
            progress_bar=False,
            merge_clusters_thr=0.01,
            logging_level=None,
            samples_early_stop=3,
            n_jobs=1)

        embeddings_features, expected_cluster_ids, expected_topics, texts = get_test_data(df=intent_dataset)
        clusters_ids = s_density.fit_predict(embeddings_features)
        cluster_topics = s_density.get_topics(texts, threshold=0.25)

        # Perform asserts
        self.check_model_attributes(cluster_instance=s_density)
        self.check_predicted_data(predicted_clusters_ids=clusters_ids, expected_cluster_ids=expected_cluster_ids)
        self.check_topics_labeling(predicted_topics=cluster_topics, expected_topics=expected_topics)

