import os
import numpy as np
import pandas as pd
import pytest
from nlp.propagation import LabelExpander, Propagator
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from nlp.ner import load_ner
from tests.config import MODELS_PATH
import torch
from nlp.pt.dataset import DataCollator, TextDataset
from nlp.pt.model import IntentClassifier
from nlp.pt.fitter import TransformersFitter


@pytest.fixture
def data_text_classification():

    texts = \
        [
         'Hi  not sure whether this question is correct for the option chosen Can you tell me if I had warranty',
         "Hi just wondering if I can get insurance on my phone as I 've just cracked the screen It 's 42",
         "hello my has damage hello my phone 's broken how can i get help",
         'Hi I need to report a berevement please however i only want to cancel the mobile phone not the broadband',
         "Hi My mum 's mobile phone account was with you phone number 72 and she died on 9 th Feb.",
         'Hi could you tell me if my phone is covered for accidental damage name nickname',
         'Hi I was just talking to one of your colleagues and my battery died on my phone I was discussing my',
         'I hope you can help I ’m not sure this is right department I need to claim for a crack on my phone insurance',
         'Good afternoon I am wondering if my tablet is insured ',
         'I would like to cancel a  only contract in my husbands name as he has sadly passed away recently   84',
         'Hi my husband passed away last year and had in error deleted some things on his phone',
         'hi this is my wife phone she died last may and i need to cancel this contract ',
         "Hi name I am name son  Unfortunately  died in December 2019 and we have only just noticed that",
         'This is my late husbands phone I just want to report and close account as he sadly passed away on 2 February',
         'Hi  I am calling to arrange to close an account my aunt name has recently passed away She had an mobile',
         'I need to cancel this account and number which belongs to my father in law as he sadly passed away last week',
         'Hi how do I check if I have insurance thanks',
         'Hi I have damage cover on my my phone what does this cover please 48',
         'Hi I sadly damaged my h9 any and needed a new one very quickly',
         'I am the executor who died on March 29. He had a phone with you 44 I informed you about death',
         "Good afternoon I was managing my Nana's account as she was 94 years old Unfortunately she passed away",
         'Hi I was wondering if i have insurance cover as my screen has stopped working to touch',
         'Hi put my phone in the wash this morning so now got a dead phone and hoping to upgrade',
         'Hi name my husband passed away last year and not sure if his phone or sim is still on this account',
         'Hi I have lifetime guarantee with your company, my phone has stopped charging and its only new 15 months',
         'Hello how do I insure my phone Thanks Both phones on my contract',
         "Hi I recently upgraded my phones and received an email with a breakdown",
         'Hello I ’m wondering if my phone is still insuranced as when I got my phone last year',
         "I 've just got my latest bill and I can not see if I 've got insurance  macey 06",
         'Hello i need to report the death of our mum please thanks',
         'Hi I would like to know if my phone is insured or if not how to insure Hi shubham yes that ’s right',
         'Hi I just want know about the insurance Yes',
         'Hello this is the account for my grandmother She has died I would like to cancel the account and stop paying',
         "My phone died it would not charge and I 'm due an upgrade yes",
         'I was just on the line to someone from upgrades and the phone went dead i was on the line',
         'hi would you please be able to tell me if I have phone insurance many thanks',
         'Hi I recently bought beat solo pros off you guys with a monthly price and they have all of a sudden stopped',
         'This line was used for my mum who passed away on Thursday I would therefore like to cancel this contract but',
         'Hi this is my moms mobile and unfortunately she has passed away Could you please advise me how to cancel',
         'Have I got insurance cover for damage for my oppo phone Ok',
        ]
    intents = ['a', 'a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'b', 'b', 'b',
               'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'b', 'a']

    return texts, intents


@pytest.fixture
def data_classifier_partitions(data_text_classification):
    texts, intents = data_text_classification
    # Build dataset (assign redundant columns for previous verification)
    utterances_df = pd.DataFrame({'text': texts, 'intent': intents})
    utterances_df['expected_intent'] = utterances_df['intent']
    utterances_df['labels'] = utterances_df['intent']

    # Assign random -1 labels, 50% of the data will have 'unlabeled' intents
    index_labeled, index_unlabeled = train_test_split(np.array(range(0, len(intents))),
                                                      stratify=intents,
                                                      test_size=0.5,
                                                      random_state=100)
    utterances_df.loc[index_unlabeled, 'intent'] = 'NO_LABEL'

    # Map intents: string to int
    intent2idx = {k: i for i, k in enumerate(utterances_df[utterances_df['intent'] != 'NO_LABEL'].groupby(
        ['intent']).first().reset_index()['intent'].values)}
    intent2idx['NO_LABEL'] = -1
    utterances_df['label'] = utterances_df['intent'].map(intent2idx)

    # initial set up to build labeled(gold) and unlabeled dataset
    df_labeled = utterances_df[utterances_df['label'] != -1].copy().reset_index(drop=True)
    df_unlabelled = utterances_df[utterances_df['label'] == -1].copy().reset_index(drop=True)

    # get class setup data
    n_classes = df_labeled['label'].nunique()
    class_weights = np.ones(n_classes)

    # Split golden dataset for training
    X_train, X_test, y_train, y_test = train_test_split(df_labeled,
                                                        df_labeled['label'].values,
                                                        test_size=0.5,
                                                        shuffle=True,
                                                        stratify=df_labeled['label'].values,
                                                        random_state=100)

    return X_train, X_test, y_train, y_test, df_labeled, df_unlabelled, \
        utterances_df, intent2idx, n_classes, class_weights


@pytest.fixture
def data_clustering():
    X = np.array([[3, 3, 3],
                  [3.5, 3, 3.1],
                  [2.5, 3.3, 3.2],
                  [-3, -3, -3],
                  [-3.5, -2.7, -3.1],
                  [-2.8, -3, -3.2],
                  [-3, 3, 3],
                  [-3.5, 2.7, 3.1],
                  [-2.8, 3, 3.2]])

    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_gaps = np.array([9, 9, 9, 3, 3, 3, 6, 6, 6])

    X_test = np.array([[3.0, 3.2, 3.1],
                       [-3.1, -2.8, -3.1],
                       [-2.9, 2.9, 3.1],
                       [4, 3.2, 3.1]])

    y_expected = np.array([0, 1, 2, -1])

    centroids_expected = [[0.56172264,  0.58268327,  0.58231336],
                          [-0.5878911, -0.55131096, -0.58881760],
                          [-0.5878911,  0.55131096,  0.58881760]]

    return X, y, X_test, y_expected, y_gaps, centroids_expected


def test_label_expander(data_clustering):

    X, y, X_test, y_expected, y_gaps, centroids_expected = data_clustering

    # Test propagation with clusters without gaps
    expander = LabelExpander(threshold=0.99, candidates=1, progress_bar=False)
    expander.fit(X, y)
    assert len(expander.centroids) == len(np.unique(y))
    assert np.allclose(expander.centroids, centroids_expected)
    y_hat = expander.transform(X_test)
    assert all(y_hat == y_expected)

    # Test non consecutive labels
    expander_b = LabelExpander(threshold=0.99, candidates=1, progress_bar=False)
    expander_b.fit(X, y_gaps)
    assert len(expander_b.centroids) == len(np.unique(y_gaps))
    assert np.allclose(expander_b.centroids, centroids_expected)
    y_hat_b = expander.transform(X_test)
    assert all(y_hat_b == y_expected)


def get_tokenizer_max_length(tokenizer, texts):
    tokens = tokenizer(texts, padding=True, truncation=True, add_special_tokens=False,
                       return_tensors="np", max_length=500)

    lens = np.count_nonzero(tokens['attention_mask'], axis=1)  # todo implement in scripts?

    return int(np.quantile(np.array(lens), 0.5))  # max 62, excluding 112 and 88 max sizes


@pytest.mark.usefixtures("clean_dir", "random_config")
class TestPropagator:

    if MODELS_PATH['roberta-base'] is None or MODELS_PATH['ner-nlp'] is None or MODELS_PATH['intent-classifier'] \
            is None:
        pytest.skip(F"Label propagation needs to setup the corresponding pretrained models: "
                    F" roberta-base: {MODELS_PATH['roberta-base']}"
                    F" ner-nlp (spacy model): {MODELS_PATH['ner-nlp']},"
                    F" intent-classifier: {MODELS_PATH['intent-classifier']} ")
    # Label propagation needs a pre-trained intent classifier. The model ...
    seed = 100
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    intent2idx = {'a': 0, 'b': 1, 'NO_LABEL': -1}
    n_classes = 2
    class_weights = np.array([1.0, 1.0])
    training_params = {
                       'n_epochs': 1,
                       'batch_size': 64,
                       'lr': 0.00005,
                       'seed': 7
                       }
    model_params = {
                    'model_name': MODELS_PATH['roberta-base'],
                    'dropout_rate': 0.9,
                    'n_outputs': 2,
                    }
    max_length = 20

    @staticmethod
    def check_save(model_path):
        assert os.path.isfile(model_path)

    @staticmethod
    def check_train(history, expected_epochs, expected_ratio_labelled, best_metric):
        expected_columns = ['iter', 'ratio_labelled', 'conf_all', 'conf_labelled', 'pct_change',
                            'train_all_loss', 'train_gold_loss']
        assert isinstance(history, pd.DataFrame)
        assert all([column in expected_columns for column in history.columns])
        assert len(history) == expected_epochs
        assert history['ratio_labelled'].iloc[-1] == 1.0
        assert np.allclose(history.ratio_labelled.values, expected_ratio_labelled)
        assert (best_metric != 0) and (best_metric != np.inf)
        # according linear/exponential approach

    @staticmethod
    def check_inference(fitter, test_loader, y_test):
        predictions_probs = fitter.predict(test_loader=test_loader)
        predictions_labels = np.argmax(predictions_probs, axis=1)
        assert isinstance(predictions_probs, np.ndarray)
        assert len(predictions_labels) == len(y_test)
        assert all([predictions_label in set(y_test) for predictions_label in predictions_labels])
        assert all(predictions_labels != -1)

    def check_propagator(self, propagator, expected_label_size, gold_labels):
        # Check structure
        expected_attributes_propagator = ['weights', 'labels', 'device', 'class_weights']
        for expected_attribute in expected_attributes_propagator:
            assert hasattr(propagator, expected_attribute)
        assert isinstance(propagator.weights, np.ndarray)
        assert isinstance(propagator.weights, np.ndarray)
        assert len(propagator.labels) == expected_label_size
        assert len(propagator.weights) == len(propagator.labels)

        # Check expected outputs
        # Check gold labels/weights has not been modified
        assert all(propagator.labels[:len(gold_labels)] == gold_labels)
        assert all(propagator.weights[:len(gold_labels)] == 1)
        # check gold labels/weights values range
        assert all(propagator.weights >= 0) and all(propagator.weights <= 1)
        assert all([label in self.intent2idx.values() for label in propagator.labels])
        assert all(propagator.labels != -1)
        isinstance(propagator.class_weights, np.ndarray)
        assert len(propagator.class_weights) == len(self.class_weights)
        assert not (np.allclose(propagator.class_weights, self.class_weights))  # check weights change with training
        assert not (np.allclose(propagator.class_weights, np.array([0.0, 0.0])))

    @staticmethod
    def check_fitter(fitter):
        # LB has fitter dependencies
        expected_attributes_fitter = ['model', 'device', 'best_metric']
        for expected_attribute in expected_attributes_fitter:
            assert hasattr(fitter, expected_attribute)
        assert isinstance(fitter, TransformersFitter)
        assert isinstance(fitter.model, IntentClassifier)

    def _get_model_components(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_params['lr'])
        tokenizer = RobertaTokenizerFast.from_pretrained(MODELS_PATH['roberta-base'])
        criterion = torch.nn.CrossEntropyLoss()
        return optimizer, tokenizer, criterion

    def _get_pretrained_models(self):
        ner, tag2id, id2tag = load_ner(MODELS_PATH['ner-nlp'])
        intent_classifier = IntentClassifier(label2idx=self.intent2idx, **self.model_params)
        intent_classifier.to(self.device)
        return ner, tag2id, id2tag, intent_classifier

    @staticmethod
    def get_data_loaders(tokenizer, texts_train, labels_train, texts_test, labels_test, n_classes, device, ner,
                         tag2id, max_length, class_weights, batch_size):
        # Training Loader
        dataset = TextDataset(data=texts_train,
                              labels=labels_train,
                              weights=np.ones(len(labels_train)),
                              class_weights=class_weights,
                              n_classes=n_classes,
                              device=device,
                              only_labelled=True)

        collate_fn = DataCollator(tokenizer, nlp=ner, tag2id=tag2id, ner=False, max_length=max_length)

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
        return train_loader, test_loader, collate_fn

    def _load_label_propagation_dependencies(self, texts_train, labels_train, texts_test, labels_test, model_path):

        # Load models
        ner, tag2id, id2tag, model = self._get_pretrained_models()

        # Load model components
        optimizer, tokenizer, criterion = self._get_model_components(model)

        # Load data loaders
        _, test_loader, collate_fn = self.get_data_loaders(tokenizer=tokenizer,
                                                           texts_train=texts_train,
                                                           labels_train=labels_train,
                                                           texts_test=texts_test,
                                                           labels_test=labels_test,
                                                           n_classes=self.model_params['n_outputs'],
                                                           device=self.device,
                                                           ner=ner,
                                                           tag2id=tag2id,
                                                           max_length=self.max_length,
                                                           class_weights=self.class_weights,
                                                           batch_size=self.training_params['batch_size'])

        # Load fitter
        fitter = TransformersFitter(model=model,
                                    device=self.device,
                                    loss=criterion,
                                    optimizer=optimizer,
                                    scheduler=None,
                                    validation_scheduler=True,
                                    step_scheduler=False,
                                    log_path='log.txt',
                                    verbose=True)

        fitter.load(model_path)

        return fitter, collate_fn, tokenizer, test_loader

    def test_exponential(self, data_classifier_partitions):

        # load data
        X_train, X_test, y_train, y_test, df_labeled, df_unlabelled, df_utterances, _, _, _ = data_classifier_partitions

        # load pretrained intent classifier, their components and testing data.
        fitter, collate_fn, tokenizer, test_loader = self._load_label_propagation_dependencies(
            texts_train=X_train['text'].values,
            labels_train=y_train,
            texts_test=X_test['text'].values,
            labels_test=y_test,
            model_path=MODELS_PATH['intent-classifier'])

        # Performs label propagation
        save_checkpoint_label_propagator = 'label_propagation_exponential/checkpoint.bin'

        propagator_params = {'n_epochs': 2,
                             'batch_size': 32,
                             'device': self.device,
                             'policy': 'exponential',  # linear
                             'seed': self.seed}

        propagator = Propagator(gold_corpus=X_train['text'].values,
                                gold_labels=y_train,
                                n_classes=self.n_classes,
                                expansion_policy=propagator_params['policy'],
                                tokenizer=tokenizer,
                                collator=collate_fn,
                                batch_size=propagator_params['batch_size'],
                                device=propagator_params['device'])

        history_lp = propagator.fit(corpus=df_unlabelled['text'].values,
                                    n_times=propagator_params['n_epochs'],
                                    fitter=fitter,  # fitter contains the loaded/trained model
                                    val_loader=test_loader,
                                    callbacks=None)

        fitter.save(save_checkpoint_label_propagator)

        # Check fitter structure (the reason: propagator affects the intent model and fitter object)
        self.check_fitter(fitter=fitter)

        # Check propagator
        self.check_propagator(propagator=propagator,
                              expected_label_size=len(df_unlabelled) + len(X_train),
                              gold_labels=y_train)

        # Check train asserts (test each new epoch inject new data with a constant size)
        self.check_train(history=history_lp, expected_epochs=propagator_params['n_epochs'],
                         expected_ratio_labelled=np.array([0.46667, 1.0]),
                         best_metric=fitter.best_metric)

        # Check model inference in trained model
        self.check_inference(fitter=fitter, test_loader=test_loader, y_test=y_test)

        # Check save
        self.check_save(model_path=save_checkpoint_label_propagator)

        # Check loaded model works (is able to performs an inference)
        fitter_loaded, _, _, _ = self._load_label_propagation_dependencies(
            texts_train=X_train['text'].values,
            labels_train=y_train,
            texts_test=X_test['text'].values,
            labels_test=y_test,
            model_path=save_checkpoint_label_propagator)
        self.check_inference(fitter=fitter_loaded, test_loader=test_loader, y_test=y_test)

    def test_training_linear(self, data_classifier_partitions):

        # load data
        X_train, X_test, y_train, y_test, df_labeled, df_unlabelled, df_utterances, _, _, _ = data_classifier_partitions

        # load pretrained intent classifier, their components and testing data.
        fitter, collate_fn, tokenizer, test_loader = self._load_label_propagation_dependencies(
            texts_train=X_train['text'].values,
            labels_train=y_train,
            texts_test=X_test['text'].values,
            labels_test=y_test,
            model_path=MODELS_PATH['intent-classifier'])

        # Performs label propagation
        save_checkpoint_label_propagator = 'label_propagation_linear/checkpoint.bin'

        propagator_params = {'n_epochs': 2,
                             'batch_size': 32,
                             'device': self.device,
                             'policy': 'linear',  # linear
                             'seed': self.seed}

        propagator = Propagator(gold_corpus=X_train['text'].values,
                                gold_labels=y_train,
                                n_classes=self.n_classes,
                                expansion_policy=propagator_params['policy'],
                                tokenizer=tokenizer,
                                collator=collate_fn,
                                batch_size=propagator_params['batch_size'],
                                device=propagator_params['device'])

        history_lp = propagator.fit(corpus=df_unlabelled['text'].values,
                                    n_times=propagator_params['n_epochs'],
                                    fitter=fitter,  # fitter contains the loaded/trained model
                                    val_loader=test_loader,
                                    callbacks=None)

        fitter.save(save_checkpoint_label_propagator)

        # Check fitter structure (the reason: propagator affects the intent model and fitter object)
        self.check_fitter(fitter=fitter)

        # Check propagator
        self.check_propagator(propagator=propagator,
                              expected_label_size=len(df_unlabelled) + len(X_train),
                              gold_labels=y_train)

        # Check train asserts (test each new epoch inject new data increasing with a constant ratio)
        self.check_train(history=history_lp, expected_epochs=propagator_params['n_epochs'],
                         expected_ratio_labelled=np.array([0.7, 1.0]),
                         best_metric=fitter.best_metric)

        # Check model inference in trained model
        self.check_inference(fitter=fitter, test_loader=test_loader, y_test=y_test)

        # Check save
        self.check_save(model_path=save_checkpoint_label_propagator)

        # Check loaded model works (is able to performs an inference)
        fitter_loaded, _, _, _ = self._load_label_propagation_dependencies(
                                texts_train=X_train['text'].values,
                                labels_train=y_train,
                                texts_test=X_test['text'].values,
                                labels_test=y_test,
                                model_path=save_checkpoint_label_propagator)
        self.check_inference(fitter=fitter_loaded, test_loader=test_loader, y_test=y_test)

