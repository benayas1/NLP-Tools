import os.path
import pickle
import pytest
import torch
import os
from nlp.pt.model import IntentClassifier
from tests.config import MODELS_PATH
from transformers import RobertaTokenizerFast


@pytest.fixture
def text_data():
    data = ['Hello I am the first text test',
            'They are meny uses cases to execute']
    return data


@pytest.mark.usefixtures('text_data', 'clean_dir')
class TestIntentClassifier:

    if MODELS_PATH['roberta-base'] is None:
        pytest.skip(F"No roberta-base model found")

    # Define common configuration variables
    model_save_path = 'model_config.p'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def check_model_structure(model):
        expected_parameters = ['label2idx', 'idx2label', 'transformer', 'return_encodings', 'return_output',
                               'training', 'model_name']
        assert all([hasattr(model, expected_parameter) for expected_parameter in expected_parameters])
        assert isinstance(model, IntentClassifier)

    def _get_tokens(self, model, text_data):
        model.eval()
        tokenizer = RobertaTokenizerFast.from_pretrained(MODELS_PATH['roberta-base'])

        tokens = tokenizer(text_data,
                           is_split_into_words=False,
                           return_offsets_mapping=False,
                           padding=True,
                           truncation=True,
                           return_tensors='pt',
                           max_length=50)

        inputs_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        inputs_ids = inputs_ids.squeeze().to(self.device).int()
        attention_mask = attention_mask.squeeze().to(self.device).int()
        return inputs_ids, attention_mask

    def check_model_inference(self, model, model_params, text_data):

        inputs, masks = self._get_tokens(model, text_data)

        # Check forward output
        output_forward = model(input_ids=inputs, attention_mask=masks)
        processed_documents, logits_len = output_forward.shape
        assert isinstance(output_forward, torch.Tensor)
        assert processed_documents == len(text_data)
        assert logits_len == model_params['n_outputs']

        # Check encode output
        output_encoded = model.encode(input_ids=inputs, attention_mask=masks).cpu() # .numpy() ->  with torch.no_grad():
        processed_documents, encodings = output_encoded.shape
        assert isinstance(output_encoded, torch.Tensor)
        assert processed_documents == len(text_data)
        assert encodings == 768

    def check_model_config_save(self, model):

        expected_params = {'class_name': str,
                           'return_output': bool,
                           'return_encodings': bool,
                           'n_outputs': int,
                           'dropout_rate': float,
                           'model_name': str,
                           'label2idx': dict,
                           'idx2label': dict}

        model.save(path=self.model_save_path)
        # Test saving config
        assert os.path.isfile(self.model_save_path)
        # Test loading config
        model_config = pickle.load(open(self.model_save_path, "rb"))
        for attribute_name, attribute_type in expected_params.items():
            assert attribute_name in model_config.keys()
            assert isinstance(model_config[attribute_name], attribute_type)

    def test_model(self, text_data):
        # Load data
        intent2idx = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        model_params = {'model_name': MODELS_PATH['roberta-base'],
                        'dropout_rate': 0.3,
                        'n_outputs': 4,
                        }
        model = IntentClassifier(label2idx=intent2idx, **model_params)
        model.to(self.device)

        self.check_model_structure(model)
        self.check_model_inference(model, model_params, text_data)
        self.check_model_config_save(model)

    def test_model_loaded(self, text_data):
        if MODELS_PATH['intent-classifier'] is None:
            pytest.skip(F"No pre trained intent-classifier model found")

        intent2idx = {'a': 1, 'b': 2}
        model_params = {'model_name': MODELS_PATH['roberta-base'],
                        'dropout_rate': 0.3,
                        'n_outputs': 2,
                        }

        model = IntentClassifier(label2idx=intent2idx, **model_params)
        model.to(self.device)
        # In order to detect if old versions are deprecated
        checkpoint = torch.load(MODELS_PATH['intent-classifier'])
        model.load_state_dict(checkpoint['model_state_dict'])

        self.check_model_structure(model)
        self.check_model_inference(model, model_params, text_data)
        self.check_model_config_save(model)
