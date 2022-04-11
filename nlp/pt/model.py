import torch
from transformers import AutoModel, AutoConfig
import pickle
import numpy as np
from typing import Union, Dict
import json


def load_model(source: Union[str, Dict]):
    if isinstance(source, str):
        if source[-4:].lower() == 'json':
            params = json.load(open(source, "rb"))
        else:
            params = pickle.load(open(source, "rb"))
    else:
        params = source

    assert 'class_name' in params, 'Class Name must be included in params'
    class_name = params['class_name']
    del params['class_name']

    try:
        class_name = globals()[class_name]
    except KeyError:
        raise KeyError(f'Class {class_name} does not exist')
    model = class_name(**params)
    return model


class StorableModel():

    def save(self, path):
        raise NotImplementedError()


# Model with classifier layers on top of RoBERTa
class IntentClassifier(torch.nn.Module, StorableModel):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 from_config: bool = False,
                 dropout_rate: float = 0.3,
                 n_outputs: int = 10,
                 return_output: bool = True,
                 return_encodings: bool = False,
                 label2idx: dict = None,
                 idx2label: dict = None):
        super(IntentClassifier, self).__init__()

        # Encoder
        if from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(config)
        else:
            self.transformer = AutoModel.from_pretrained(model_name)

        # Classifier
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 256)
        self.bn1 = torch.nn.LayerNorm(256)
        self.act1 = torch.nn.ReLU()

        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(256, n_outputs)

        # Return options
        self.return_output = return_output
        self.return_encodings = return_encodings

        # Mapping
        if label2idx is not None:
            self.label2idx = label2idx
            if idx2label is not None:
                assert np.all([label in label2idx for label in idx2label.values()]), "Mappings do not match"
                assert np.all([idx in idx2label for idx in label2idx.values()]), "Mappings do not match"
                self.idx2label = idx2label
            else:
                self.idx2label = {v: k for k, v in label2idx.items()}
        else:
            if idx2label is not None:
                self.idx2label = idx2label
                self.label2idx = {v: k for k, v in idx2label.items()}
            else:
                print("Warning: No mapping provided, won't be able to save the model")

    def forward(self, input_ids, attention_mask):
        encodings = self.encode(input_ids, attention_mask)

        if self.return_encodings and not self.return_output:
            return encodings

        x = self.d1(encodings)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.d2(x)
        x = self.l2(x)

        if self.return_encodings:
            return x, encodings

        return x

    def encode(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return x['pooler_output']

    def save(self, path):
        params = {'class_name': self.__class__.__name__,
                  'return_output': self.return_output,
                  'return_encodings': self.return_encodings,
                  'n_outputs': self.l2.out_features,
                  'dropout_rate': self.d2.p,
                  'model_name': self.transformer.name_or_path,
                  'label2idx': self.label2idx,
                  'idx2label': self.idx2label}

        pickle.dump(params, open(path, "wb"))

    @property
    def model_name(self):
        return self.transformer.name_or_path


class IntentClassifierNER(torch.nn.Module, StorableModel):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 from_config: bool = False,
                 dropout_rate: float = 0.3,
                 n_intents: int = 10,
                 n_ner: int = 20,
                 return_output: bool = True,
                 return_encodings: bool = False,
                 label2idx: dict = None,
                 idx2label: dict = None):
        super(IntentClassifierNER, self).__init__()

        # Encoder
        if from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(config)
        else:
            self.transformer = AutoModel.from_pretrained(model_name)

        # Intent Classifier
        self.ic_d1 = torch.nn.Dropout(dropout_rate)
        self.ic_l1 = torch.nn.Linear(768, 256)
        self.ic_bn1 = torch.nn.LayerNorm(256)
        self.ic_act1 = torch.nn.ReLU()

        self.ic_d2 = torch.nn.Dropout(dropout_rate)
        self.ic_l2 = torch.nn.Linear(256, n_intents)

        # NER
        self.ner_d1 = torch.nn.Dropout(dropout_rate)
        self.ner_l1 = torch.nn.Linear(768, 256)
        self.ner_bn1 = torch.nn.LayerNorm(256)
        self.ner_act1 = torch.nn.ReLU()

        self.ner_d2 = torch.nn.Dropout(dropout_rate)
        self.ner_l2 = torch.nn.Linear(256, n_ner)

        # Return options
        self.return_output = return_output
        self.return_encodings = return_encodings

        # Mapping
        if label2idx is not None:
            self.label2idx = label2idx
            if idx2label is not None:
                assert np.all([label in label2idx for label in idx2label.values()]), "Mappings do not match"
                assert np.all([idx in idx2label for idx in label2idx.values()]), "Mappings do not match"
                self.idx2label = idx2label
            else:
                self.idx2label = {v: k for k, v in label2idx.items()}
        else:
            if idx2label is not None:
                self.idx2label = idx2label
                self.label2idx = {v: k for k, v in idx2label.items()}
            else:
                print("Warning: No mapping provided, won't be able to save the model")

    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.return_encodings and not self.return_output:
            return x['pooler_output']

        # Intent classifier
        x1 = self.ic_d1(x['pooler_output'])
        x1 = self.ic_l1(x1)
        x1 = self.ic_bn1(x1)
        x1 = self.ic_act1(x1)
        x1 = self.ic_d2(x1)
        x1 = self.ic_l2(x1)

        # NER
        x2 = self.ner_d1(x['last_hidden_state'])
        x2 = self.ner_l1(x2)
        x2 = self.ner_bn1(x2)
        x2 = self.ner_act1(x2)
        x2 = self.ner_d2(x2)
        x2 = self.ner_l2(x2)

        return x1, x2  # x1 shape is (N, n_intents) and x2 (N, n_tokens, n_entities)

    def encode(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return x['pooler_output']

    def save(self, path):
        params = {'class_name': self.__class__.__name__,
                  'return_output': self.return_output,
                  'return_encodings': self.return_encodings,
                  'n_intents': self.ic_l2.out_features,
                  'n_ner': self.ner_l2.out_features,
                  'dropout_rate': self.d2.p,
                  'model_name': self.transformer.name_or_path,
                  'label2idx': self.label2idx,
                  'idx2label': self.idx2label}

        pickle.dump(params, open(path, "wb"))

    @property
    def model_name(self):
        return self.transformer.name_or_path


class IntentClassifierNERShared(torch.nn.Module, StorableModel):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 from_config: bool = False,
                 dropout_rate: float = 0.3,
                 n_intents: int = 10,
                 n_ner: int = 20,
                 return_output: bool = True,
                 return_encodings: bool = False,
                 label2idx: dict = None,
                 idx2label: dict = None):
        super(IntentClassifierNERShared, self).__init__()

        # Encoder
        if from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(config)
        else:
            self.transformer = AutoModel.from_pretrained(model_name)

        # Intent Classifier
        self.ic_d1 = torch.nn.Dropout(dropout_rate)
        self.ic_l1 = torch.nn.Linear(768, 256)
        self.ic_bn1 = torch.nn.LayerNorm(256)
        self.ic_act1 = torch.nn.ReLU()

        self.ic_d2 = torch.nn.Dropout(dropout_rate)
        self.ic_l2 = torch.nn.Linear(256, n_intents)

        # NER
        self.ner_d1 = torch.nn.Dropout(dropout_rate)
        self.ner_l1 = torch.nn.Linear(768, 256)
        self.ner_bn1 = torch.nn.LayerNorm(256)
        self.ner_act1 = torch.nn.ReLU()

        self.ner_d2 = torch.nn.Dropout(dropout_rate)
        self.ner_l2 = torch.nn.Linear(256, n_ner)

        # Return options
        self.return_output = return_output
        self.return_encodings = return_encodings

        # Mapping
        if label2idx is not None:
            self.label2idx = label2idx
            if idx2label is not None:
                assert np.all([label in label2idx for label in idx2label.values()]), "Mappings do not match"
                assert np.all([idx in idx2label for idx in label2idx.values()]), "Mappings do not match"
                self.idx2label = idx2label
            else:
                self.idx2label = {v: k for k, v in label2idx.items()}
        else:
            if idx2label is not None:
                self.idx2label = idx2label
                self.label2idx = {v: k for k, v in idx2label.items()}
            else:
                print("Warning: No mapping provided, won't be able to save the model")

    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.return_encodings and not self.return_output:
            return x['pooler_output']

        # Intent classifier branch
        x1 = self.ic_d1(x['pooler_output'])
        x1 = self.ic_l1(x1)
        x1 = self.ic_bn1(x1)
        x1 = self.ic_act1(x1)  # shape is (N, 64)

        # NER branch
        x2 = self.ner_d1(x['last_hidden_state'])
        x2 = self.ner_l1(x2)
        x2 = self.ner_bn1(x2)
        x2 = self.ner_act1(x2)  # shape is (N, n_batch, 64)

        # Share parameters
        x1_tmp = x1 * torch.mean(x2, dim=2)
        x2_tmp = x2 * torch.unsqueeze(x1, dim=2)

        # IC head
        x1 = self.ic_d2(x1_tmp)
        x1 = self.ic_l2(x1)

        # NER head
        x2 = self.ner_d2(x2_tmp)
        x2 = self.ner_l2(x2)

        return x1, x2  # x1 shape is (N, n_intents) and x2 (N, n_tokens, n_entities)

    def encode(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return x['pooler_output']

    def save(self, path):
        params = {'class_name': self.__class__.__name__,
                  'return_output': self.return_output,
                  'return_encodings': self.return_encodings,
                  'n_intents': self.ic_l2.out_features,
                  'n_ner': self.ner_l2.out_features,
                  'dropout_rate': self.d2.p,
                  'model_name': self.transformer.name_or_path,
                  'label2idx': self.label2idx,
                  'idx2label': self.idx2label}

        pickle.dump(params, open(path, "wb"))

    @property
    def model_name(self):
        return self.transformer.name_or_path


class IntentClassifierNERShared_v3_1(torch.nn.Module, StorableModel):
    """
    Implement last version of IC-NER MultiTask network.
    Interactions between both connections have been optimised:
        · Fixed the IC branch, where now we introduce a trainable combination of both IC and NER data.
        This branch is potentially close to its optimal architecture.
        · To perform a similar interaction of data in the NER branch, it is mandatory to expand IC batch dimension,
        adapt it to NER FF layer scale and transpose it.

    """
    def __init__(self, 
                 model_name: str ='roberta-base',
                 from_config: bool = False,
                 dropout_rate: float = 0.3, 
                 n_intents: int = 10,
                 ic_emb: int = 64,
                 n_ner: int = 20,
                 ner_emb: int = 64,
                 return_output: bool = True, 
                 return_encodings: bool = False,
                 label2idx: dict = None,
                 idx2label: dict = None):
        super(IntentClassifierNERShared, self).__init__()
        
        # Encoder
        if from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(config)
        else:
            self.transformer = AutoModel.from_pretrained(model_name)

        #Layers
        ## IC
        self.ic_d1 = torch.nn.Dropout(dropout_rate)
        self.ic_l1 = torch.nn.Linear(768, ic_emb)
        self.ic_bn1 = torch.nn.LayerNorm(ic_emb)
        self.ic_act1 = torch.nn.ReLU()

        self.ic_shared_ner = torch.nn.Linear(1, ner_emb, dtype = torch.double)

        self.ic_l2 = torch.nn.Linear(ner_emb, 1)
        self.ic_fl1 = torch.nn.Flatten()
        self.ic_l3 = torch.nn.Linear(ic_emb, n_intents)

        ##NER
        self.ner_d1 = torch.nn.Dropout(dropout_rate)
        self.ner_l1 = torch.nn.Linear(768, ner_emb)
        self.ner_bn1 = torch.nn.LayerNorm(ner_emb)
        self.ner_act1 = torch.nn.ReLU()

        self.ner_l2 = torch.nn.Linear(ic_emb, n_ner, dtype = torch.double)

        # Return options
        self.return_output=return_output
        self.return_encodings=return_encodings

        # Mapping
        if label2idx is not None:
            self.label2idx = label2idx
            if idx2label is not None:
                assert np.all([ label in label2idx for label in idx2label.values()]), "Mappings do not match"
                assert np.all([ idx in idx2label for idx in label2idx.values()]), "Mappings do not match"
                self.idx2label = idx2label
            else:
                self.idx2label = {v:k for k,v in label2idx.items()}
        else:
            if idx2label is not None:
                self.idx2label = idx2label
                self.label2idx = {v:k for k,v in idx2label.items()}
            else:
                print("Warning: No mapping provided, won't be able to save the model")       
        
    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.return_encodings and not self.return_output:
            return x['pooler_output']

        # Intent classifier branch
        x_ic = self.ic_d1(x['pooler_output'])
        x_ic = self.ic_l1(x_ic)
        x_ic = self.ic_bn1(x_ic)
        x_ic = self.ic_act1(x_ic)


        # NER branch
        x_ner = self.ner_d1(x['last_hidden_state'])
        x_ner = self.ner_l1(x_ner)
        x_ner = self.ner_bn1(x_ner)
        x_ner = self.ner_act1(x_ner)

        # Shared information
        x_ic_to_ner = torch.unsqueeze(x_ic, axis = -1)
        x_ic_to_ner = self.ic_shared_ner(x_ic_to_ner)
        x_ic_to_ner = torch.transpose(x_ic_to_ner, dim0 = -2, dim1 = -1)
        x_ner_to_ic = torch.mean(x_ner, axis = 1, keepdim = True)

        #IC output
        x_ic = torch.matmul(torch.unsqueeze(x_ic, axis = -1), x_ner_to_ic)
        x_ic = self.ic_l2(x_ic)
        x_ic = self.ic_fl1(x_ic)
        ic_output = self.ic_l3(x_ic)

        # NER output
        x_ner = torch.matmul(x_ner, x_ic_to_ner)
        ner_output = self.ner_l2(x_ner)

        return ic_output, ner_output

    def encode(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return x['pooler_output']

    def save(self, path):
        params = {'class_name':self.__class__.__name__,
                  'return_output':self.return_output,
                  'return_encodings':self.return_encodings,
                  'n_intents':self.ic_l2.out_features,
                  'n_ner': self.ner_l2.out_features,
                  'dropout_rate':self.d2.p,
                  'model_name':self.transformer.name_or_path,
                  'label2idx':self.label2idx,
                  'idx2label':self.idx2label}

        pickle.dump(params, open(path, "wb"))

    @property
    def model_name(self):
        return self.transformer.name_or_path


class IntentClassifierNERShared_v3_2(torch.nn.Module, StorableModel):
    """
    Implement last version of IC-NER MultiTask network.
    Interactions between both connections have been optimised:
    · Fixed the IC branch, where now we introduce a trainable combination of both IC and NER data.
    This branch is potentially close to its optimal architecture.
    · To perform a similar interaction of data in the NER branch, it is mandatory to reshape,
    expand and adapt it to NER output.

    """
    def __init__(self, 
                 model_name: str ='roberta-base',
                 from_config: bool = False,
                 dropout_rate: float = 0.3, 
                 n_intents: int = 10,
                 ic_emb: int = 64,
                 n_ner: int = 20,
                 ner_emb: int = 64,
                 return_output: bool = True, 
                 return_encodings: bool = False,
                 label2idx: dict = None,
                 idx2label: dict = None):
        super(IntentClassifierNERShared, self).__init__()
        
        # Encoder
        if from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(config)
        else:
            self.transformer = AutoModel.from_pretrained(model_name)

        #Layers
        ## IC
        self.ic_d1 = torch.nn.Dropout(dropout_rate)
        self.ic_l1 = torch.nn.Linear(768, ic_emb)
        self.ic_bn1 = torch.nn.LayerNorm(ic_emb)
        self.ic_act1 = torch.nn.ReLU()

        self.ic_shared_ner_1 = torch.nn.Linear(ic_emb, ner_emb, dtype = torch.double)
        self.ic_shared_ner_2 = torch.nn.Linear(1, n_ner, dtype = torch.double)

        self.ic_l2 = torch.nn.Linear(ner_emb, 1)
        self.ic_fl1 = torch.nn.Flatten()
        self.ic_l3 = torch.nn.Linear(ic_emb, n_intents)

        ##NER
        self.ner_d1 = torch.nn.Dropout(dropout_rate)
        self.ner_l1 = torch.nn.Linear(768, ner_emb)
        self.ner_bn1 = torch.nn.LayerNorm(ner_emb)
        self.ner_act1 = torch.nn.ReLU()

        # Return options
        self.return_output=return_output
        self.return_encodings=return_encodings

        # Mapping
        if label2idx is not None:
            self.label2idx = label2idx
            if idx2label is not None:
                assert np.all([ label in label2idx for label in idx2label.values()]), "Mappings do not match"
                assert np.all([ idx in idx2label for idx in label2idx.values()]), "Mappings do not match"
                self.idx2label = idx2label
            else:
                self.idx2label = {v:k for k,v in label2idx.items()}
        else:
            if idx2label is not None:
                self.idx2label = idx2label
                self.label2idx = {v:k for k,v in idx2label.items()}
            else:
                print("Warning: No mapping provided, won't be able to save the model")       
        
    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if self.return_encodings and not self.return_output:
            return x['pooler_output']

        # Intent classifier branch
        x_ic = self.ic_d1(x['pooler_output'])
        x_ic = self.ic_l1(x_ic)
        x_ic = self.ic_bn1(x_ic)
        x_ic = self.ic_act1(x_ic)


        # NER branch
        x_ner = self.ner_d1(x['last_hidden_state'])
        x_ner = self.ner_l1(x_ner)
        x_ner = self.ner_bn1(x_ner)
        x_ner = self.ner_act1(x_ner)

        # Shared information
        x_ic_to_ner = self.ic_shared_ner_1(x_ic)
        x_ic_to_ner = torch.unsqueeze(x_ic_to_ner, axis = -1)
        x_ic_to_ner = self.ic_shared_ner_2(x_ic_to_ner)
        x_ner_to_ic = torch.mean(x_ner, axis = 1, keepdim = True)

        #IC output
        x_ic = torch.matmul(torch.unsqueeze(x_ic, axis = -1), x_ner_to_ic)
        x_ic = self.ic_l2(x_ic)
        x_ic = self.ic_fl1(x_ic)
        ic_output = self.ic_l3(x_ic)

        # NER output
        ner_output = torch.matmul(x_ner, x_ic_to_ner)

        return ic_output, ner_output

    def encode(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return x['pooler_output']

    def save(self, path):
        params = {'class_name':self.__class__.__name__,
                  'return_output':self.return_output,
                  'return_encodings':self.return_encodings,
                  'n_intents':self.ic_l2.out_features,
                  'n_ner': self.ner_l2.out_features,
                  'dropout_rate':self.d2.p,
                  'model_name':self.transformer.name_or_path,
                  'label2idx':self.label2idx,
                  'idx2label':self.idx2label}

        pickle.dump(params, open(path, "wb"))

    @property
    def model_name(self):
        return self.transformer.name_or_path