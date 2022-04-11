import os
import pytest
import nlp.preprocessing as pp


@pytest.fixture(scope='function')
def prep_pipe():
    config = {
        'appos': False,
        'slang': False,
        'sep_digit_text': False,
        'emoticons': False,
        'emoticons_del': False,
        'eol_remove': False,
        'eol_replace': False,
        'html': False,
        'url': False,
        'email': False,
        'proper_noun': False,
        'repeated_chars': False,
        'single_char': False,
        'punct': False,
        'number': False,
        'extra_space': False,
        'stop_words': False,
        'lemmas': False
    }
    return pp.Pipeline(config)


def test_lemmas(prep_pipe):
    # Test lemma configuration to check the processed text return the text with lemmas
    prep_pipe.lemmas = True
    expected_lemmas = {'are': 'be',
                       'been': 'be',
                       'feets': 'feet',
                       'has': 'have',
                       'running': 'run',
                       'children': 'child',
                       'boys': 'boy',
                       'girls': 'girl',
                       'persons': 'person',
                       'teeth': 'tooth'
                       }
    texts = [
             "we are in your feets right now",
             "she has been running",
             "Michael has 4 beautiful children",
             "they are many persons without teeth"
             ]
    expected_texts = [
        f"we {expected_lemmas['are']} in your {expected_lemmas['feets']} right now ",
        f"she {expected_lemmas['has']} {expected_lemmas['been']} {expected_lemmas['running']} ",
        f"Michael {expected_lemmas['has']} 4 beautiful {expected_lemmas['children']} ",
        f"they {expected_lemmas['are']} many {expected_lemmas['persons']} without {expected_lemmas['teeth']} ",
                     ]
    transformed_texts = prep_pipe.fit_transform(texts)
    assert len(texts) == len(transformed_texts)
    assert isinstance(transformed_texts, list)
    for expected_text, transformed_text in zip(expected_texts, transformed_texts):
        assert isinstance(expected_text, str)
        assert expected_text == transformed_text


def test_preprocessing_masking(prep_pipe):
    # test masking implemented on private method mask()
    prep_pipe.proper_noun = 'default_name'
    prep_pipe.email = 'default_email'
    prep_pipe.url = 'default_url'
    prep_pipe.phone_number = 'default_phone_number'
    prep_pipe.number = 'default_number'

    texts = ["My name is Kevin Bacon, my phone number is 9999999996 ends with 6",
             "We're David Jones and Nick Smith the www.web_page_link.com account holders",
             "Hey James Johnson did you talked with Maria Johnson?",
             "Testing if email@gmail.com and e-mail@company.ar gets replaced 2 times",
             "David number: is 5 888888888, email: name@domain.com, website: www.jl.com"]

    expected_texts = \
        [f"My name is {prep_pipe.proper_noun} {prep_pipe.proper_noun} , my phone number is {prep_pipe.phone_number} "
         f"ends with {prep_pipe.number} ",
         f"We're {prep_pipe.proper_noun} {prep_pipe.proper_noun} and {prep_pipe.proper_noun} {prep_pipe.proper_noun} "
         f"the {prep_pipe.url} account holders ",
         f"Hey {prep_pipe.proper_noun} {prep_pipe.proper_noun} did you talked with {prep_pipe.proper_noun} "
         f"{prep_pipe.proper_noun} ? ",
         f"Testing if {prep_pipe.email} and {prep_pipe.email} gets replaced {prep_pipe.number} times ",
         f"{prep_pipe.proper_noun} number : is {prep_pipe.number} {prep_pipe.phone_number} , email : {prep_pipe.email} "
         f", website : {prep_pipe.url} "]

    transformed_texts = prep_pipe.fit_transform(texts)

    assert len(transformed_texts) == len(texts)
    assert isinstance(transformed_texts, list)
    for expected_text, transformed_text in zip(expected_texts, transformed_texts):
        assert isinstance(expected_text, str)
        assert expected_text == transformed_text


def test_tokenizer(prep_pipe):
    texts = ["Hello this is a text    without tokenization",
             "The test price is just 5$, the 3% of all I've ",
             "What are you doing? Come now!! Let's go"]
    texts_tokenized_expected = [
        ['Hello', 'this', 'is', 'a', 'text', '   ', 'without', 'tokenization'],
        ['The', 'test', 'price', 'is', 'just', '5', '$', ',', 'the', '3', '%', 'of', 'all', 'I', "'ve"],
        ['What', 'are', 'you', 'doing', '?', 'Come', 'now', '!', '!', 'Let', "'s", 'go']]

    # Check KeyError raise when model is not fitted
    with pytest.raises(KeyError):
        prep_pipe.tokenize()

    # Fit the pipeline and test tokenized text are as expected
    prep_pipe.fit(texts)
    text_tokenized = prep_pipe.tokenize()
    assert isinstance(text_tokenized, list)
    assert len(text_tokenized) == len(texts)
    for index, text_tokenized_expected in enumerate(texts_tokenized_expected):
        assert text_tokenized_expected == text_tokenized[index]


@pytest.mark.usefixtures("clean_dir", "prep_pipe")
class TestSaverLoader:
    # Class for testing the loading and saving process considering two uses cases, using fit and fit_transform methods
    model_path = 'pre_processing_model.pk'

    texts = [
        "A test fixture is an environment used to consistently test some item, device, or piece of software. Test "
        "fixtures can be found when testing electronics, software and physical devices.",
        "A software test fixture sets up a system for the software testing process by initializing it, thereby "
        "satisfying any preconditions the system may have.[1] For example, the Ruby on Rails web framework uses YAML "
        "to initialize a database with known parameters before running a test.[2] This allows for tests to be "
        "repeatable, which is one of the key features of an effective test framework.[1]",
        "The advantage of a test fixture is that it allows for tests to be repeatable since each test is always "
        "starting with the same setup. Test fixtures also ease test code design by allowing the developer to separate "
        "methods into different functions and reuse each function for other tests. Further, test fixtures preconfigure "
        "tests into a known initial state instead of working with whatever was left from a previous test run. "
        "A disadvantage is that it could lead to duplication of test fixtures if using in-line setup"
            ]

    expected_texts = [
        'A test fixture is an environment used to consistently test some item , device , or piece of software . Test '
        'fixtures can be found when testing electronics , software and physical devices . ',
        'A software test fixture sets up a system for the software testing process by initializing it , thereby '
        'satisfying any preconditions the system may have.[1 ] For example , the Ruby on Rails web framework uses YAML '
        'to initialize a database with known parameters before running a test.[2 ] This allows for tests to be '
        'repeatable , which is default_number of the key features of an effective test framework.[1 ] ',
        'The advantage of a test fixture is that it allows for tests to be repeatable since each test is always '
        'starting with the same setup . Test fixtures also ease test code design by allowing the developer to separate '
        'methods into different functions and reuse each function for other tests . Further , test fixtures '
        'preconfigure tests into a known initial state instead of working with whatever was left from a previous test '
        'run . A disadvantage is that it could lead to duplication of test fixtures if using in - line setup '
    ]

    def setup_pipeline(self, prep_pipe, mode='fit'):
        prep_pipe.proper_noun = 'default_name'
        prep_pipe.email = 'default_email'
        prep_pipe.url = 'default_url'
        prep_pipe.phone_number = 'default_phone_number'
        prep_pipe.number = 'default_number'
        prep_pipe.sep_digit_text = True
        if mode == 'fit':
            prep_pipe.fit(self.texts)
        else:
            prep_pipe.fit_transform(self.texts)
        return prep_pipe

    def save_model(self, prep_pipe: pp, mode='fit'):
        prep_pipe = self.setup_pipeline(prep_pipe, mode)
        prep_pipe.save(path=self.model_path)
        return prep_pipe

    def check_loaded_model(self, prep_pipe_loaded, prep_pipe_baseline):
        # Compare baseline model in memory against the loaded model
        assert isinstance(prep_pipe_loaded, pp.Pipeline)

        # Check model configuration compare
        for expected_key,  expected_value in prep_pipe_baseline.config.items():
            assert hasattr(prep_pipe_loaded, expected_key)
            assert getattr(prep_pipe_loaded, expected_key) == expected_value

        # Check model transformations per document
        for key, text in enumerate(prep_pipe_loaded.docs):
            # Check text is kept without filter (only cleaning + spacy pipe)
            assert text.text == self.texts[key]
            # Check text is transformed as expected
            assert prep_pipe_loaded.text(key) == self.expected_texts[key]

    def test_model_saving_fit(self, prep_pipe):
        _ = self.save_model(prep_pipe, mode='fit')
        assert os.path.isfile(self.model_path)

    def test_model_saving_fit_transform(self, prep_pipe):
        self.save_model(prep_pipe, mode='fit_transform')
        assert os.path.isfile(self.model_path)

    def test_model_loader_fit(self, prep_pipe):
        # Save model after performing the fit() method
        prep_pipe_baseline = self.save_model(prep_pipe, mode='fit')
        prep_pipe.load(path=self.model_path)
        self.check_loaded_model(prep_pipe_loaded=prep_pipe, prep_pipe_baseline=prep_pipe_baseline)
        # Test transform method
        assert prep_pipe.transform() == self.expected_texts

    def test_model_loader_fit_transform(self, prep_pipe):
        # Save model after performing the fit_transform() method
        prep_pipe_baseline = self.save_model(prep_pipe, mode='fit_transform')
        prep_pipe.load(path=self.model_path)
        self.check_loaded_model(prep_pipe_loaded=prep_pipe, prep_pipe_baseline=prep_pipe_baseline)
        # Test transform method cannot be called after call fit() method
        with pytest.raises(KeyError):
            prep_pipe.transform()
        # Test fit transform method:
        assert prep_pipe.fit_transform(self.texts) == self.expected_texts


def test_appos_look_up(prep_pipe):
    prep_pipe.appos = True
    text = ["testing if these things can't or cant be replaced",
            "another appos isnt this another example?"]
    transformed = prep_pipe.fit_transform(text)

    assert "can't" not in transformed[0]
    assert "cant" not in transformed[0]
    assert "isnt" not in transformed[1]
    assert "can not" in transformed[0]
    assert "is not" in transformed[1]


def test_slang(prep_pipe):
    prep_pipe.slang = True
    text = ["testing if this is awsm",
            "we need this done asap"]
    transformed = prep_pipe.fit_transform(text)

    assert "awsm" not in transformed[0]
    assert "asap" not in transformed[1]
    assert "awesome" in transformed[0]
    assert "as soon as possible" in transformed[1]


def test_separate_digit_text(prep_pipe):
    prep_pipe.sep_digit_text = True
    text = ["There are 2seats for 3guys",
            "I want iphone10"]
    transformed = prep_pipe.fit_transform(text)

    assert "2seats" not in transformed[0]
    assert "3guys" not in transformed[0]
    assert "2 seats" in transformed[0]
    assert "3 guys" in transformed[0]
    assert "iphone10" in transformed[1]


def test_url(prep_pipe):
    prep_pipe.url = True
    text = ['testing if https://docs.python.org/3/library/multiprocessing.html gets removed',
            'another http://www.casetophono.com/2019/07/chillin.html example']
    transformed = prep_pipe.fit_transform(text)

    assert "https" not in transformed[0]
    assert "http" not in transformed[1]


def test_email(prep_pipe):
    prep_pipe.email = True
    text = ['testing if email@gmail.com gets removed',
            'another email@email.es example']
    transformed = prep_pipe.fit_transform(text)
    assert "email@gmail.com" not in transformed[0]
    assert "email@email.es" not in transformed[1]


def test_proper_noun(prep_pipe):
    prep_pipe.proper_noun = True
    text = ["My name is Kevin Bacon, I am the account holder",
            "hey James Johnson how are you doing"]
    transformed = prep_pipe.fit_transform(text)
    assert "Kevin Bacon" not in transformed[0]
    assert "James Johnson" not in transformed[1]


def test_repeated_chars(prep_pipe):
    prep_pipe.repeated_chars = True
    text = ['I am verrrrrry happppyyyyy']
    transformed = prep_pipe.fit_transform(text)[0]
    assert 'verry' in transformed
    assert 'happyy' in transformed


def test_eol(prep_pipe):
    prep_pipe.eol_remove = True
    text = ['this it a test\n I want to test\r\n again \n and another test \ntesting and more\nand more\r\ntesting']
    transformed = prep_pipe.fit_transform(text)[0]
    assert '\n' not in transformed
    assert '\r\n' not in transformed
    assert '. ' not in transformed


def test_html(prep_pipe):
    prep_pipe.html = True
    text = ['this <spam> is a test <aref image=www> of html tags ']
    transformed = prep_pipe.fit_transform(text)[0]
    assert '<' not in transformed
    assert '>' not in transformed
    assert '<spam>' not in transformed
    assert '< spam >' not in transformed
    assert '< aref image=www >' not in transformed


