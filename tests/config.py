import os

# If any path in MODELS_PATH is None the test routine will skip the related unit test
TEST_MODEL_PATH = "C:\\Users\\Jonathan_Espinosa\\Projects\\cc-analytics-nlp-framework\\test_models"
TEMP_TEST_FILE = 'tmp'
MODELS_PATH = {
               'glove': os.path.join(TEST_MODEL_PATH, 'glove', 'glove.6B.300d.txt'),
               'bert-base-uncased': os.path.join(TEST_MODEL_PATH, 'bert-base-uncased'),
               'roberta-base': os.path.join(TEST_MODEL_PATH, 'roberta-base'),
               'translation-models':  os.path.join(TEST_MODEL_PATH, 'Helsinki-NLP'),
               'opus-mt-en-de': os.path.join(TEST_MODEL_PATH, 'opus-mt-en-de'),
               'paraphrase-albert-small-v2': os.path.join(TEST_MODEL_PATH, 'paraphrase-albert-small-v2'),
               'intent-classifier': os.path.join(TEST_MODEL_PATH, 'intent-classifier', 'best-checkpoint.bin'),
               'ner-nlp': os.path.join(TEST_MODEL_PATH, 'ner-nlp'),
               'w2v': os.path.join(TEST_MODEL_PATH, 'w2v', 'w2v.wordvectors')
               }

