import numpy as np
import pandas as pd
import pytest
from transformers import RobertaTokenizerFast
from nlp.ner import load_ner
from tests.config import MODELS_PATH
import torch
from nlp.pt.dataset import DataCollator, TextDataset


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
               'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'b', 'b', 'b', 'a', 'NO_LABEL', 'NO_LABEL',
               'NO_LABEL', 'NO_LABEL']
    weights = [0.5, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5]
    intent2idx = {'a': 0, 'b': 1, 'NO_LABEL': -1}
    df_texts = pd.DataFrame({'text': texts, 'intent': intents, 'weight': weights})
    df_texts['label'] = df_texts['intent'].map(intent2idx)
    n_classes = 2
    class_weights = np.ones(n_classes)

    return df_texts, intent2idx, n_classes, class_weights


class TestTextDataset:

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def check_data_structure(text, label, weight):
        assert isinstance(text, str)
        assert isinstance(label, np.int64)
        assert isinstance(weight, float)

    def test_default_values_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        df_texts = df_texts[~df_texts.label.isin([-1])]
        dataset = TextDataset(data=df_texts.text.values,
                              labels=df_texts.label.values,
                              weights=None,
                              class_weights=None,
                              n_classes=None,
                              device=self.device,
                              only_labelled=True)
        assert len(dataset.class_weights) == len(class_weights)
        assert len(dataset) == len(df_texts[~df_texts.label.isin([-1])])
        index = 0
        text, label, weight = dataset[index]
        text == df_texts.text.iloc[index]
        self.check_data_structure(text, label, weight)
        assert text == df_texts.text.iloc[index]
        assert label == df_texts.label.iloc[index]
        assert weight == 1

    def test_filtering_unlabeled_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        dataset = TextDataset(data=df_texts.text,
                              labels=df_texts.label,
                              weights=df_texts.weight,
                              class_weights=class_weights,
                              n_classes=n_classes,
                              device=self.device,
                              only_labelled=True)
        assert len(dataset) == len(df_texts[~df_texts.label.isin([-1])])
        index = 0
        text, label, weight = dataset[index]
        self.check_data_structure(text, label, weight)
        assert text == df_texts.text.iloc[index]
        assert label == df_texts.label.iloc[index]
        assert weight == df_texts.weight.iloc[index]

    def test_only_labelled_false_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        dataset = TextDataset(data=df_texts.text,
                              labels=df_texts.label,
                              weights=df_texts.weight,
                              class_weights=class_weights,
                              n_classes=n_classes,
                              device=self.device,
                              only_labelled=False)
        assert isinstance(dataset, TextDataset)
        assert len(dataset) == len(df_texts)
        index = 0
        text = dataset[index]
        assert isinstance(text, str)
        assert text == df_texts.text.iloc[index]

    def test_custom_class_weight_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        df_texts = df_texts[~df_texts.label.isin([-1])]
        dataset = TextDataset(data=df_texts.text.values,
                              labels=df_texts.label.values,
                              weights=df_texts.weight.values,
                              class_weights=np.array([0.5, 0.5]),
                              n_classes=n_classes,
                              device=self.device,
                              only_labelled=True)
        assert len(dataset) == len(df_texts[~df_texts.label.isin([-1])])

        index = 0
        text, label, weight = dataset[index]
        self.check_data_structure(text, label, weight)
        assert text == df_texts.text.iloc[index]
        assert label == df_texts.label.iloc[index]
        assert weight == 0.25


class TestDataCollator:

    max_length = 20
    batch_size = 8
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if MODELS_PATH['roberta-base'] is None or MODELS_PATH['ner-nlp'] is None:
        pytest.skip(F"Collator needs to setup the corresponding pretrained models")

    def get_collator_dependencies(self, text, labels, weights, n_classes):
        # Training Loader
        dataset = TextDataset(data=text,
                              labels=labels,
                              weights=weights,
                              class_weights=None,
                              n_classes=n_classes,
                              device=self.device,
                              only_labelled=True)
        ner, tag2id, id2tag = load_ner(MODELS_PATH['ner-nlp'])
        tokenizer = RobertaTokenizerFast.from_pretrained(MODELS_PATH['roberta-base'])
        return ner, tag2id, id2tag, tokenizer, dataset

    @staticmethod
    def unpack_dataloader(data_loader):
        # Unpack dataloader
        for step, data in enumerate(data_loader):
            input_ids = data['x']['input_ids']
            attention_mask = data['x']['attention_mask']
            y = data['y']
            w = data['w']
        return input_ids, attention_mask, y, w

    @staticmethod
    def check_inputs_and_weights(input_ids, attention_mask, weights, df_texts):
        # Test inputs and weights
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert all(weights.detach().cpu().numpy() == df_texts.weight)
        assert len(input_ids) == len(df_texts)
        assert len(attention_mask) == len(df_texts)

    def test_unique_task_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        df_texts = df_texts[~df_texts.label.isin([-1])]
        _, tag2id, id2tag, tokenizer, dataset = self.get_collator_dependencies(text=df_texts.text,
                                                                               labels=df_texts.label,
                                                                               weights=df_texts.weight,
                                                                               n_classes=n_classes)
        collate_fn = DataCollator(tokenizer=tokenizer, nlp=None, tag2id=tag2id, ner=False, max_length=self.max_length)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=len(df_texts),
                                                  collate_fn=collate_fn,
                                                  num_workers=2,
                                                  shuffle=False)

        # Unpack data
        input_ids, attention_mask, y, w = self.unpack_dataloader(data_loader)

        # Test inputs and weights
        self.check_inputs_and_weights(input_ids=input_ids, attention_mask=attention_mask, weights=w, df_texts=df_texts)

        # Test labels
        assert isinstance(y, torch.Tensor)
        assert all(y.detach().cpu().numpy() == df_texts.label.values)

    def test_multitask_case(self, data_text_classification):
        df_texts, intent2idx, n_classes, class_weights = data_text_classification
        df_texts = df_texts[~df_texts.label.isin([-1])]
        ner, tag2id, id2tag, tokenizer, dataset = self.get_collator_dependencies(text=df_texts.text,
                                                                                 labels=df_texts.label,
                                                                                 weights=df_texts.weight,
                                                                                 n_classes=n_classes)

        collate_fn = DataCollator(tokenizer=tokenizer, nlp=ner, tag2id=tag2id, ner=True, max_length=self.max_length)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=len(df_texts),
                                                  collate_fn=collate_fn,
                                                  num_workers=2,
                                                  shuffle=False)
        # Unpack dataloader
        input_ids, attention_mask, y, w = self.unpack_dataloader(data_loader)

        # Test inputs and weights
        self.check_inputs_and_weights(input_ids=input_ids, attention_mask=attention_mask, weights=w, df_texts=df_texts)

        # Test labels
        y_intent, y_ner = y
        assert isinstance(y, list)
        assert isinstance(y_intent, torch.Tensor)
        assert isinstance(y_ner, torch.Tensor)
        number_documents, max_ner_size = y_ner.shape
        assert number_documents == len(df_texts)
        assert max_ner_size == self.max_length
        assert all(y_intent.detach().cpu().numpy() == df_texts.label.values)
