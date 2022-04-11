from nlp.augmentation import augment_intent, BackTranslation, SynonymReplacement, OneOf, Augmenter, \
    SwitchAugmentation, ReplaceAugmentation, InsertAugmentation, Sequence, SynonymReplacementFast
from Levenshtein import distance
from collections import Counter
import pandas as pd
import pytest
from tests.config import MODELS_PATH

# Is recommended to apply the replacement,insert and switch replacement in text with similar size to guarantee minimal
# amount changes will work as expected.


@pytest.fixture(scope='module')
def df_intent():
    df_intent = pd.DataFrame({
        'text': ["Hello there can you hear me",
                 "go could you let me knw please",
                 "I have no service and I want it",
                 "why can not i see the details",
                 "Thanks perfect I will call you",
                 "Hi I am Jon how I can help you"],
        'intent': ["intent_a", "intent_b", "intent_a", "intent_b", "intent_a", "intent_c"]})
    return df_intent


@pytest.fixture
def troubled_texts():
    return ['yes', 'no', 'perfect', 'right', 'correcto', '', ' ', "", " ", '909', 'empty', "! ok #", "?"]


@pytest.fixture(scope='module')
def texts_reference(df_intent):
    texts = df_intent['text'].tolist()
    return texts


@pytest.fixture(scope='module')
def augmentation_list():
    n_mistakes = 2
    min_mistakes = 1

    switch_augmentation = SwitchAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)
    replace_augmentation = ReplaceAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)
    insert_augmentation = InsertAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)

    augmentation_list = [switch_augmentation, replace_augmentation, insert_augmentation]

    return augmentation_list


@pytest.mark.usefixtures("random_config")
def test_one_of(texts_reference, augmentation_list):
    one_of = OneOf(augmenters=augmentation_list, p=[0.2]*(len(augmentation_list)-1))
    text_augmented = one_of.augment(texts=texts_reference)

    # Check object type
    assert isinstance(one_of, Augmenter)

    # Check object attributes
    assert hasattr(one_of, 'p')
    assert hasattr(one_of, 'augmenters')

    # Check probability array
    assert sum(one_of.p) == 1

    # Check augmented batch size
    assert len(text_augmented) == len(texts_reference)

    one_of_default = OneOf(augmenters=augmentation_list, p=None)
    text_augmented_default = one_of_default.augment(texts=texts_reference)

    # Check probability array
    assert sum(one_of_default.p) == 1

    # Check augmented batch size
    assert len(text_augmented_default) == len(texts_reference)


@pytest.mark.usefixtures("random_config")
def test_sequence(texts_reference, augmentation_list):

    sequence_augmenters = Sequence(augmenters=augmentation_list)

    text_augmented = sequence_augmenters.augment(texts=texts_reference)

    # Check object type
    assert isinstance(sequence_augmenters, Augmenter)

    # Check object attributes
    assert hasattr(sequence_augmenters, 'p')
    assert hasattr(sequence_augmenters, 'augmenters')

    # Check returned text is different from reference
    assert all([text_reference != text_augmented[idx] for idx, text_reference in enumerate(texts_reference)])

    # Check amount transformations
    assert len(sequence_augmenters.augmenters) == len(augmentation_list)


@pytest.mark.usefixtures("random_config")
def test_switch_augmentation(texts_reference):
    # please use odd values to avoid getting the same string example idx[2,3] : 1- Hello, 2- Helol, 3- Hello
    n_mistakes = [7, 5]
    min_mistakes = [2, 3]
    test_cases = zip(n_mistakes, min_mistakes)

    for n_mistakes, min_mistakes in test_cases:
        switch_augmentation = SwitchAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)
        texts_augmented = switch_augmentation.augment(texts=texts_reference)

        # Check object type
        assert isinstance(switch_augmentation, Augmenter)

        for i in range(0, len(texts_reference)):
            text_reference_non_spaces = texts_reference[i].replace(' ', '')
            text_augmented_non_spaces = texts_augmented[i].replace(' ', '')
            length_delta_non_spaces = len(text_augmented_non_spaces) - len(text_augmented_non_spaces)
            length_delta = len(texts_augmented[i]) - len(texts_reference[i])

            # Check size integrity
            assert length_delta_non_spaces == 0

            # Check augmented share same amount of characters
            assert Counter(text_reference_non_spaces) == Counter(text_augmented_non_spaces)

            # Check max-min changes (levenshtein penalize 2 each switch +1 deletion and +1 insertion)
            # Sometime two switch can compute just 1 or 0 levenshtein score because changes are computed in spaces
            levenshtein_distance = distance(texts_reference[i], texts_augmented[i])
            # assert levenshtein_distance >= min_mistakes  # we cannot guarantee changes in words with repeated letters
            # like hello, see, call etc. Where switching some positions will produce the same word
            assert levenshtein_distance <= n_mistakes * 2


@pytest.mark.usefixtures("random_config")
def test_replace_augmentation(texts_reference):
    n_mistakes = [5, 7, 9]
    min_mistakes = [1, 1, 1]
    test_cases = zip(n_mistakes, min_mistakes)

    for n_mistakes, min_mistakes in test_cases:
        replace_augmentation = ReplaceAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)
        texts_augmented = replace_augmentation.augment(texts=texts_reference)

        # Check object type
        assert isinstance(replace_augmentation, Augmenter)

        for i in range(0, len(texts_reference)):
            length_reference = len(texts_reference[i])
            length_augmented = len(texts_augmented[i])
            length_delta = length_augmented - length_reference

            # Check replacements size
            assert length_delta == 0

            # Check text keep similarity (each levenshtein replacement counts +1)
            levenshtein_distance = distance(texts_reference[i], texts_augmented[i])
            assert levenshtein_distance >= min_mistakes
            assert levenshtein_distance <= n_mistakes


@pytest.mark.usefixtures("random_config")
def test_insert_augmentation(texts_reference):
    n_mistakes = [1, 2, 3]
    min_mistakes = [1, 1, 1]
    test_cases = zip(n_mistakes, min_mistakes)

    for n_mistakes, min_mistakes in test_cases:
        insert_augmentation = InsertAugmentation(n_mistakes=n_mistakes, min_mistakes=min_mistakes)
        texts_augmented = insert_augmentation.augment(texts=texts_reference)

        # Check object type
        assert isinstance(insert_augmentation, Augmenter)

        for i in range(0, len(texts_reference)):
            length_reference = len(texts_reference[i])
            length_augmented = len(texts_augmented[i])
            length_delta = length_augmented - length_reference

            # Check inserts
            assert length_augmented >= length_reference
            assert length_delta >= min_mistakes
            assert length_delta <= n_mistakes

            # Check text keep similarity (each levenshtein insertion counts +1)
            levenshtein_distance = distance(texts_reference[i], texts_augmented[i])
            assert levenshtein_distance >= min_mistakes
            assert levenshtein_distance <= n_mistakes


@pytest.mark.usefixtures("random_config")
def test_augment_intent(augmentation_list, df_intent):

    augmenter = augmentation_list[0]

    assert len(augment_intent(df=df_intent, augmenter=None, lower_bound=0, upper_bound=3)) == 3

    assert len(augment_intent(df=df_intent, augmenter=augmenter, lower_bound=10, upper_bound=100)) == 10

    assert len(augment_intent(df=df_intent, augmenter=augmenter, lower_bound=0, upper_bound=100)) == len(df_intent)


@pytest.mark.usefixtures("random_config")
def test_synonym_replacement(texts_reference):
    if MODELS_PATH['glove'] is None:
        pytest.skip(f"Missing embeddings_path attribute for SynonymReplacement")

    synonym_replacement = SynonymReplacement(model_type='glove',
                                             embeddings_path=MODELS_PATH['glove'])

    augmented_texts = synonym_replacement.augment(texts_reference)
    print(augmented_texts)

    # Test dimensions
    assert len(augmented_texts) == len(texts_reference)
    # Test data types
    assert all([isinstance(augmented_text, str) for augmented_text in augmented_texts])

    for i in range(len(texts_reference)):
        # Test augmented text replace unless 1 word
        assert len(set(texts_reference[i].split()).difference(set(augmented_texts[i].split()))) >= 1

    assert True


@pytest.mark.usefixtures("random_config")
def test_back_translation(texts_reference, troubled_texts):

    if MODELS_PATH['translation-models'] is None:
        pytest.skip(f"Missing models_path path attribute for BackTranslation")

    language = 'de'
    back_translation = BackTranslation(language=language,
                                       models_path=MODELS_PATH['translation-models'])
    text_translations = back_translation.translate(texts=texts_reference,
                                                   model=back_translation.model_src2tmp,
                                                   tokenizer=back_translation.tokenizer_src2tmp,
                                                   language=language)
    text_augmented = back_translation.augment(texts=texts_reference)

    # Check object type
    assert isinstance(back_translation, Augmenter)

    # Check batch size integrity
    assert len(texts_reference) == len(text_augmented)

    # Check translation and source language are different
    assert back_translation.src != language

    for i in range(len(texts_reference)):
        # Check translation datatype
        assert isinstance(text_translations[i], str)

        # Check translation is not empty
        assert len(text_translations[i].replace(" ", "")) > 0

        # Check augmented is not empty
        assert len(text_augmented[i].replace(" ", "")) > 0

        # Check augmented datatype
        assert isinstance(text_augmented[i], str)

        # Check translation return a different string
        assert text_translations[i] != texts_reference[i]

    # Test trouble cases translation
    text_translations_trouble = back_translation.translate(
        texts=troubled_texts,
        model=back_translation.model_src2tmp,
        tokenizer=back_translation.tokenizer_src2tmp,
        language=language)
    # Test if return same amount documents
    assert len(text_translations_trouble) == len(troubled_texts)
    # Test augment text is different
    assert len(set(text_translations_trouble + troubled_texts)) > len(troubled_texts)

    # Test trouble cases augmentation -> translation and backtranslation
    text_augmented_trouble = back_translation.augment(texts=troubled_texts)
    # Test if return same amount documents
    assert len(text_augmented_trouble) == len(troubled_texts)
    # Test augment text is different
    assert len(set(text_augmented_trouble + troubled_texts)) > len(troubled_texts)


def test_synonym_replacement_fast(df_intent, troubled_texts):

    if MODELS_PATH['w2v'] is None:
        pytest.skip(f"Missing models_path w2v required for class SynonymReplacementFast")

    synonyms = SynonymReplacementFast(MODELS_PATH['w2v'], aug_p=0.3, device='cpu')

    augmented_text = synonyms.augment(texts=df_intent.text)
    assert isinstance(augmented_text, list) # Check data type
    assert len(augmented_text) == len(df_intent) # Check size consistency
    assert all([isinstance(sentence, str) for sentence in augmented_text]) # Check data types
    assert len(set(augmented_text+df_intent.text.to_list())) > len(df_intent) # Check translation provide new text

    augmented_troubled_text = synonyms.augment(texts=troubled_texts)
    assert isinstance(augmented_troubled_text, list)  # Check data type
    assert len(augmented_troubled_text) == len(troubled_texts)  # Check size consistency
    assert all([isinstance(sentence, str) for sentence in augmented_troubled_text])  # Check data types
    assert len(set(augmented_troubled_text + troubled_texts)) > len(troubled_texts)  #


if __name__ == "__main__":
    pass
