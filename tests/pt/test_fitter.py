import os
import pytest
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast
from sklearn.metrics import accuracy_score, f1_score
from nlp.ner import load_ner
from tests.config import MODELS_PATH
from nlp.pt.dataset import DataCollator, TextDataset
from nlp.pt.model import IntentClassifier
from nlp.pt.fitter import TransformersFitter, AverageMeter
from nlp.pt.loss import LabelSmoothingCrossEntropyLoss


@pytest.fixture
def data_text_classification():
    texts = \
        [
         'Hi I would like to know if my phone is insured or if not how to insure Hi shubham yes that ’s right',
         'Hi I just want know about the insurance Yes',
         'Hello this is the account for my grandmother She has died I would like to cancel the account and stop paying',
         "My phone died it would not charge and I 'm due an upgrade yes",
         'I was just on the line to someone from upgrades and the phone went dead i was on the line',
         'hi would you please be able to tell me if I have phone insurance many thanks',
         'Hi I recently bought beat solo pros off you guys with a monthly price and they have all of a sudden stopped',
         'This line was used for my mum who passed away on Thursday I would therefore like to cancel this contract but',
         'Hi this is my moms mobile and unfortunately she has passed away Could you please advise me how to cancel',
         'Have I got insurance cover for damage for my oppo phone Ok'
        ]
    intents = ['a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'b', 'a']
    labels = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    intent2idx = {'a': 0, 'b': 1}
    split_index = np.array([True]*6 + [False]*4)
    df = pd.DataFrame({'text': texts, 'intent': intents, 'label': labels})
    df_train = df[split_index]
    df_test = df[~split_index]
    n_classes = 2

    return df_train, df_test, intent2idx, n_classes


@pytest.fixture()
def loss_data():
    loss_data = {'loss': [0.111, 0.109, 0.106, 0.103, 0.105],
                 'batch_size': 8,
                 'expected_sum': 4.272,
                 'expected_count': 40,
                 'expected_avg': 0.1068}
    return loss_data


@pytest.mark.usefixtures('loss_data')
class TestAverageMeter:

    @staticmethod
    def build_avg_meter(loss_data):
        avg_meter = AverageMeter()
        for loss in loss_data['loss']:
            avg_meter.update(val=loss, n=loss_data['batch_size'])
        return avg_meter

    @staticmethod
    def check_attributes(avg_meter):
        expected_attributes = {'val': float, 'avg': float, 'sum': float, 'count': int}
        for expected_attribute, expected_type in expected_attributes.items():
            assert hasattr(avg_meter, expected_attribute)
            assert isinstance(getattr(avg_meter, expected_attribute), expected_type)

    def test_update(self, loss_data):
        avg_meter = self.build_avg_meter(loss_data=loss_data)
        self.check_attributes(avg_meter=avg_meter)
        assert avg_meter.val == loss_data['loss'][-1]
        assert avg_meter.sum == loss_data['expected_sum']
        assert avg_meter.count == loss_data['expected_count']
        assert avg_meter.avg == loss_data['expected_avg']

    def test_reset(self, loss_data):
        avg_meter = self.build_avg_meter(loss_data=loss_data)
        avg_meter.reset()
        self.check_attributes(avg_meter=avg_meter)
        assert avg_meter.val == 0.0
        assert avg_meter.sum == 0.0
        assert avg_meter.count == 0
        assert avg_meter.avg == 0.0


@pytest.mark.usefixtures('data_text_classification', 'clean_dir', 'random_config')
class TestTransformersFitter:

    if MODELS_PATH['roberta-base'] is None or MODELS_PATH['ner-nlp'] is None:
        pytest.skip(F"Intent classifier needs to setup the corresponding pretrained models")

    # Define common configuration variables
    save_checkpoint_intent = 'models_test_intents/last-checkpoint.bin'
    save_best_checkpoint_intent = 'models_test_intents/best-checkpoint.bin'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_texts = ['Hello I am the first text test', 'They are meny uses cases to execute']

    def _load_fitter_dependencies(self, intent2idx, training_params, n_classes):

        # Load models
        model = IntentClassifier(label2idx=intent2idx,
                                 model_name=MODELS_PATH['roberta-base'],
                                 dropout_rate=0.5,
                                 n_outputs=n_classes)
        model.to(self.device)
        ner, tag2id, id2tag = load_ner(MODELS_PATH['ner-nlp'])
        # Load model components
        tokenizer = RobertaTokenizerFast.from_pretrained(MODELS_PATH['roberta-base'])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])
        collate_fn = DataCollator(tokenizer, nlp=ner, tag2id=tag2id, ner=False, max_length=50)

        return model, tokenizer, criterion, optimizer, collate_fn

    @staticmethod
    def get_data_loaders(collate_fn, texts_train, labels_train, texts_test, labels_test, n_classes, device,
                         class_weights, batch_size):
        # Training Loader
        dataset = TextDataset(data=texts_train,
                              labels=labels_train,
                              weights=np.ones(len(labels_train)),
                              class_weights=class_weights,
                              n_classes=n_classes,
                              device=device,
                              only_labelled=True)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   collate_fn=collate_fn,
                                                   num_workers=2,
                                                   shuffle=False)

        # Validation Loader
        test_dataset = TextDataset(data=texts_test,
                                   labels=labels_test,
                                   weights=np.ones(len(labels_test)),
                                   class_weights=class_weights,
                                   n_classes=n_classes,
                                   device=device,
                                   only_labelled=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn,
                                                  num_workers=2,
                                                  shuffle=False)
        return train_loader, test_loader

    @staticmethod
    def unpack_one_batch_data(fitter, data_loader):
        for step, data in enumerate(data_loader):
            x, y, w = fitter.unpack(data=data)
            break
        return x, y, w

    @staticmethod
    def check_train_components(history, expected_epochs, best_metric, expected_metrics):
        expected_columns = ['train', 'lr', 'val'] + expected_metrics
        assert isinstance(history, pd.DataFrame)
        assert all([column in expected_columns for column in history.columns])
        assert len(history) == expected_epochs
        assert (best_metric != 0) and (best_metric != np.inf)

    @staticmethod
    def check_inference(fitter, data_loader, expected_n_classes, expected_n_predictions):
        prediction_logits = fitter.predict(test_loader=data_loader)
        number_utterances, number_classes = prediction_logits.shape
        assert isinstance(prediction_logits, np.ndarray)
        assert number_classes == expected_n_classes
        assert number_utterances == expected_n_predictions

    @staticmethod
    def check_components(fitter, criterion):
        expected_instance_attributes = {'best_metric': float, 'device': str, 'epoch': int, 'log_path': str,
                                        'verbose': bool, 'model': IntentClassifier,
                                        'optimizer': torch.optim.Adam, 'scheduler': type(None),
                                        'step_scheduler': bool, 'loss_function': type(criterion)}

        for attribute_name, attribute_type in expected_instance_attributes.items():
            assert hasattr(fitter, attribute_name)
            assert isinstance(getattr(fitter, attribute_name), attribute_type)

    @staticmethod
    def check_validation(fitter, train_loader, test_loader, expected_batch_size):
        summary_loss, calculated_metrics = fitter.validation(val_loader=train_loader, metric=None)
        assert isinstance(summary_loss.avg, float)
        assert calculated_metrics is None
        assert summary_loss.count == expected_batch_size  # len(df_train)
        # Loss and metric
        metrics = [(f1_score, {'average': 'macro'}), (accuracy_score, {})]
        metric_names = [metric[0].__name__ for metric in metrics]
        summary_loss, calculated_metrics = fitter.validation(val_loader=test_loader, metric=metrics)
        assert isinstance(summary_loss.avg, float)
        assert isinstance(calculated_metrics, list)
        assert summary_loss.count == expected_batch_size  # len(df_test)
        for value, metric_name in calculated_metrics:
            assert 0.0 <= value <= 1.0
            assert metric_name in metric_names

    def check_save(self, fitter):
        fitter.save(path=self.save_best_checkpoint_intent, verbose=True)
        assert os.path.isfile(self.save_best_checkpoint_intent)

    def check_unpack_data(self, fitter, data_loader):
        x, y, w = self.unpack_one_batch_data(fitter, data_loader)
        assert isinstance(x, dict)
        assert all([v.device.type == self.device for k, v in x.items()])
        assert isinstance(y, torch.Tensor)
        assert y.device.type == self.device
        assert isinstance(w, torch.Tensor)
        assert w.device.type == self.device

    def check_reduce_loss(self, fitter, data_loader):
        x, y, w = self.unpack_one_batch_data(fitter, data_loader)
        # Test reduce_loss with reduction (default)
        outputs = fitter.model(**x)
        loss = fitter.loss_function(outputs, y)
        loss_reduce_default = fitter.reduce_loss(loss=loss, weights=None)
        assert loss == loss_reduce_default

        # Test reduce_loss without reduction
        weight_factor = 0.5
        criterion = LabelSmoothingCrossEntropyLoss(label_smoothing=0.1,
                                                   reduction='None',  # needed to test the weighted behaviour
                                                   n_classes=2)
        loss_non_reduction = criterion(outputs, y)
        loss_reduce_custom = fitter.reduce_loss(loss=loss_non_reduction, weights=torch.tensor(np.ones(len(y)))*weight_factor)
        expected_reduce_loss = loss_non_reduction.mean() * weight_factor
        assert np.allclose(loss_reduce_custom.item(), expected_reduce_loss.item())

    def test_fit(self, data_text_classification):
        if MODELS_PATH['roberta-base'] is None:
            pytest.skip("The base model 'roberta-base' was not found, is necessary to setup the model path "
                        "configuration")

        # Load data
        df_train, df_test, intent2idx, n_classes = data_text_classification

        # Define train config for fitter
        training_params = {'n_epochs': 1,
                           'batch_size': 16,
                           'lr': 0.00005,
                           'seed': 7}
        metrics = [(f1_score, {'average': 'macro'}), (accuracy_score, {})]

        # Load fitter dependencies to init it
        model, tokenizer, criterion, optimizer, collate_fn = self._load_fitter_dependencies(intent2idx, training_params,
                                                                                            n_classes=n_classes)

        # Get loaders
        train_loader, test_loader = self.get_data_loaders(collate_fn=collate_fn,
                                                          texts_train=df_train['text'].values,
                                                          labels_train=df_train['label'].values,
                                                          texts_test=df_test['text'].values,
                                                          labels_test=df_test['label'].values,
                                                          n_classes=n_classes,
                                                          device=self.device,
                                                          class_weights=np.ones(n_classes),
                                                          batch_size=training_params['batch_size'])

        # Create fitter object for intent classifier model
        fitter = TransformersFitter(model=model,
                                    device=self.device,
                                    loss=criterion,
                                    optimizer=optimizer,
                                    scheduler=None,
                                    validation_scheduler=True,
                                    step_scheduler=False,
                                    log_path='log.txt',
                                    verbose=True)

        # Train intent classifier using the fitter object
        history = fitter.fit(train_loader=train_loader,
                             val_loader=test_loader,
                             n_epochs=training_params['n_epochs'],
                             metrics=metrics,
                             early_stopping=1,
                             early_stopping_mode='max',
                             early_stopping_alpha=0.0,
                             early_stopping_pct=0.0,
                             save_checkpoint=self.save_checkpoint_intent,
                             save_best_checkpoint=self.save_best_checkpoint_intent,
                             verbose_steps=1,
                             callbacks=None)

        # Test components
        self.check_components(fitter, criterion)

        # Test validation
        self.check_validation(fitter, train_loader, test_loader,
                              expected_batch_size=training_params['batch_size'])

        # Test inference
        self.check_inference(fitter=fitter, data_loader=test_loader,
                             expected_n_classes=n_classes,
                             expected_n_predictions=len(df_test))
        # Test save
        self.check_save(fitter)

        # Test reduce loss
        self.check_reduce_loss(fitter=fitter, data_loader=train_loader)

        # Test unpack method
        self.check_unpack_data(fitter=fitter, data_loader=test_loader)

        # Test training results
        self.check_train_components(history=history,
                                    expected_epochs=training_params['n_epochs'],
                                    best_metric=fitter.best_metric,
                                    expected_metrics=[metric[0].__name__ for metric in metrics])

    def test_load(self, data_text_classification):

        if MODELS_PATH['roberta-base'] is None or MODELS_PATH['intent-classifier'] is None:
            pytest.skip("The base models was not found, is necessary to setup the model path configuration")

        # Load testing data
        df_train, df_test, intent2idx, n_classes = data_text_classification

        # Define fit parameters
        training_params = {'n_epochs': 1,
                           'batch_size': 8,
                           'lr': 0.00005,
                           'seed': 7}

        # Load fitter dependencies
        model, tokenizer, criterion, optimizer, collate_fn = self._load_fitter_dependencies(intent2idx, training_params,
                                                                                            n_classes=n_classes)

        # Get loaders
        train_loader, test_loader = self.get_data_loaders(collate_fn=collate_fn,
                                                          texts_train=df_train['text'].values,
                                                          labels_train=df_train['label'].values,
                                                          texts_test=df_test['text'].values,
                                                          labels_test=df_test['label'].values,
                                                          n_classes=n_classes,
                                                          device=self.device,
                                                          class_weights=np.ones(n_classes),
                                                          batch_size=training_params['batch_size'])

        # criterion = torch.nn.CrossEntropyLoss()
        fitter = TransformersFitter(model=model,
                                    device=self.device,
                                    loss=criterion,
                                    optimizer=optimizer,
                                    scheduler=None,
                                    validation_scheduler=True,
                                    step_scheduler=False,
                                    log_path='log.txt',
                                    verbose=True)

        fitter.load(MODELS_PATH['intent-classifier'])

        # Test components
        self.check_components(fitter, criterion)

        # Test validation
        self.check_validation(fitter, train_loader, test_loader,
                              expected_batch_size=training_params['batch_size'])

        # Test inference
        self.check_inference(fitter=fitter, data_loader=test_loader,
                             expected_n_classes=n_classes,
                             expected_n_predictions=len(df_test))
        # Test save
        self.check_save(fitter)

        # Test reduce loss
        self.check_reduce_loss(fitter=fitter, data_loader=train_loader)

        # Test unpack method
        self.check_unpack_data(fitter=fitter, data_loader=test_loader)

