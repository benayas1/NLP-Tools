import pytest
import numpy as np
from nlp.preprocessing import Pipeline
from nlp.embeddings import LdaEmbeddings, SentenceTransformerEmbeddings, BagOfPOSEmbeddings,\
    KeyWordsEmbeddings, TransformerEmbeddings
from tests.config import MODELS_PATH


@pytest.fixture(scope="module")
def texts():
    n_repeat = 30
    texts = ['This is a test sentence about the test on spend cap keywords in the text cap',
             'And this is the second test sentence for the spend cap testing',
             'And third test we said the capitan has a new cap a little cap'] * n_repeat + \
            ['the fourth example has repeated words like dog dog dog dog dog dog',
             'the five sample has duplicated words like cat cat cat cat cat cat'] * n_repeat
    return texts


@pytest.fixture(scope='module')
def fitted_preprocessing_pipeline(texts):
    config = {
        'stop_words': True,
    }
    preprocessing_pipeline = Pipeline(config)
    preprocessing_pipeline.fit(texts)
    return preprocessing_pipeline


@pytest.fixture(scope='module')
def clean_texts(fitted_preprocessing_pipeline):
    clean_texts = fitted_preprocessing_pipeline.transform()
    return clean_texts


def test_transformer_embeddings(clean_texts, fitted_preprocessing_pipeline):

    if MODELS_PATH['paraphrase-albert-small-v2'] is None:
        pytest.skip(f"Missing model_string attribute for SentenceTransformerEmbeddings")

    expected_features = 768
    sentence_transformer_emb = SentenceTransformerEmbeddings(model_string=MODELS_PATH['paraphrase-albert-small-v2'])
    emb_matrix = sentence_transformer_emb.fit_transform(clean_texts)
    document_size, features_size = emb_matrix.shape

    # Test using List[str]] input
    # Test output type
    assert isinstance(emb_matrix, np.ndarray)
    # Test output size
    assert document_size == len(clean_texts)
    assert features_size == expected_features
    # Test value resolution
    assert isinstance(emb_matrix[0][0], np.float32)

    # Test using Pipeline input
    emb_matrix_b = sentence_transformer_emb.fit_transform(fitted_preprocessing_pipeline)
    assert isinstance(emb_matrix_b, np.ndarray)
    assert np.array_equal(emb_matrix, emb_matrix_b)


def test_sentence_transformer_embeddings(clean_texts, fitted_preprocessing_pipeline):
    if MODELS_PATH['bert-base-uncased'] is None:
        pytest.skip(f"Missing model_string attribute for SentenceTransformerEmbeddings")
    expected_features = 768
    transformer_embeddings = TransformerEmbeddings(model_string=MODELS_PATH['bert-base-uncased'])
    emb_matrix = transformer_embeddings.fit_transform(clean_texts)
    document_size, features_size = emb_matrix.shape

    # Test using List[str]] input
    # Test output type
    assert isinstance(emb_matrix, np.ndarray)
    # Test output size
    assert document_size == len(clean_texts)
    assert features_size == expected_features
    # Test value resolution
    assert isinstance(emb_matrix[0][0], np.float32)

    # Test using Pipeline input
    emb_matrix_b = transformer_embeddings.fit_transform(fitted_preprocessing_pipeline)
    assert isinstance(emb_matrix_b, np.ndarray)
    assert np.array_equal(emb_matrix, emb_matrix_b)


def test_bag_of_pos_embeddings(fitted_preprocessing_pipeline):

    # Testing POS = 'tag
    bag_pos_embeddings = BagOfPOSEmbeddings(part_of_speech='tag', progress_bar=True)
    emb_matrix = bag_pos_embeddings.fit_transform(fitted_preprocessing_pipeline)
    # Check instance attributes
    expected_attributes = ['part_of_speech', 'progress_bar', 'feature_names']
    for expected_attribute in expected_attributes:
        assert hasattr(bag_pos_embeddings, expected_attribute)
    # Check data type
    assert isinstance(emb_matrix, np.ndarray)
    # Check matrix shape
    documents_size, features_size = emb_matrix.shape
    assert documents_size == len(fitted_preprocessing_pipeline)
    assert features_size == len(bag_pos_embeddings.feature_names)
    # Check range parameters (normalized)
    assert emb_matrix.max() <= 1
    assert emb_matrix.min() >= 0
    # Check transformation return same matrix when calling more than once
    emb_matrix_b = bag_pos_embeddings.fit_transform(fitted_preprocessing_pipeline)
    assert np.allclose(emb_matrix_b, emb_matrix)

    # Testing POS = 'pos'
    bag_pos_embeddings_b = BagOfPOSEmbeddings(part_of_speech='pos', progress_bar=True)
    emb_matrix_c = bag_pos_embeddings_b.fit_transform(fitted_preprocessing_pipeline)
    # Check data type
    assert isinstance(emb_matrix_c, np.ndarray)
    # Check matrix shape
    documents_size, features_size = emb_matrix_c.shape
    assert documents_size == len(fitted_preprocessing_pipeline)
    assert features_size == len(bag_pos_embeddings_b.feature_names)
    # Check range parameters (normalized)
    assert emb_matrix_c.max() <= 1
    assert emb_matrix_c.min() >= 0

    # Check both pos tag methods produce different features
    assert not set(bag_pos_embeddings.feature_names).intersection(set(bag_pos_embeddings_b.feature_names))


def test_lda_embeddings(fitted_preprocessing_pipeline, clean_texts):
    top_k_topics = 3
    expected_documents = len(clean_texts)
    expected_instance_attributes = ['dictionary', 'n_topics', 'n_process', 'progress_bar', 'lda_model_tfidf']
    lda_embeddings = LdaEmbeddings(n_topics=top_k_topics,
                                   n_process=1,
                                   progress_bar=False)

    # Test input pipeline
    lda_embeddings.fit(fitted_preprocessing_pipeline)
    emb_matrix = lda_embeddings.transform(fitted_preprocessing_pipeline)
    # Test instance attributes
    for expected_instance_attribute in expected_instance_attributes:
        hasattr(lda_embeddings, expected_instance_attribute)
    # Test output type
    isinstance(emb_matrix, np.ndarray)
    # Test matrix dims:
    documents_size, features_size = emb_matrix.shape
    assert features_size == top_k_topics
    assert documents_size == expected_documents
    # Text topic importance weights ranges
    assert round(emb_matrix.sum()) == expected_documents
    assert emb_matrix.max() <= 1.0
    assert emb_matrix.min() >= 0.0

    # Test input tokenized text
    tokenized_text = fitted_preprocessing_pipeline.tokenize()
    emb_matrix_b = lda_embeddings.transform(tokenized_text)
    # Test result from tokenized input is almost similar to pipeline input
    assert np.allclose(emb_matrix, emb_matrix_b, rtol=1e-3)

    # Test fit_transform method results
    emb_matrix_c = lda_embeddings.fit_transform(tokenized_text)
    # Test output type
    isinstance(emb_matrix_c, np.ndarray)
    # Test matrix dims:
    documents_size_c, features_size_c = emb_matrix_c.shape
    assert features_size_c == top_k_topics
    assert documents_size_c == expected_documents


def test_keywords(clean_texts, fitted_preprocessing_pipeline):

    # MODE: binary
    embeddings_binary = KeyWordsEmbeddings(top_k=3, mode='binary', key_words=['cap', 'spend cap'])
    emb_matrix_a = embeddings_binary.fit_transform(fitted_preprocessing_pipeline)
    emb_matrix_b = embeddings_binary.fit_transform(clean_texts)
    # Compute expected dimensions
    features_size_expected = len(embeddings_binary.top_keyword_vocab) + len(embeddings_binary.key_words)
    document_size_expected = len(clean_texts)

    # Test matrix dims:
    documents_size, features_size = emb_matrix_a.shape
    assert features_size == features_size_expected
    assert documents_size == document_size_expected
    # Text feature extraction is same while input is list of text or Pipeline
    assert all((emb_matrix_a == emb_matrix_b).flatten())
    # Test data type
    isinstance(emb_matrix_a, np.ndarray)
    # Text binary feature normalization
    assert emb_matrix_a.max() == 1.0
    assert emb_matrix_a.min() == 0.0
    # Test known results:
    assert emb_matrix_a[0].mean() == 0.4285714328289032
    assert emb_matrix_a[1].mean() == 0.2857142984867096
    assert emb_matrix_a[2].mean() == 0.1428571492433548

    # MODE: COUNT
    embeddings_count = KeyWordsEmbeddings(top_k=3, mode='count', key_words=['cap', 'spend cap'])
    emb_matrix_c = embeddings_count.fit_transform(fitted_preprocessing_pipeline)

    # Text max feature normalization
    assert emb_matrix_c.max() == 1.0
    assert emb_matrix_c.min() == 0.0
    # Test known results:
    assert emb_matrix_c[0].mean() == 0.095238097012043
    assert emb_matrix_c[1].mean() == 0.0476190485060215
    assert emb_matrix_c[2].mean() == 0.0357142873108387

