import spacy
import collections
from nltk.util import ngrams


def count_words(data):
    """
    Returns a Counter with pairs word=>count for each word in the text

    Args:
        text (str):text on which processing needs to done
    Returns:
        collections.Counter: A Counter containing the word counts for each word
    """
    if isinstance(data, spacy.tokens.Doc):
        text = data.text
    else:
        text = data
    wordcount = collections.Counter(text.lower().split())
    return wordcount


def count_ngrams(data, n_gram=2):
    """
    Returns a Counter with pairs ngram=>count for each ngram in the text

    Args:
        text (str):text on which processing needs to done
        n_gram (int, optional): Length of ngrams. Defaults to 2.
    Returns:
        collections.Counter: A Counter containing the ngram counts for each ngram
    """
    if isinstance(data, spacy.tokens.Doc):
        text = data.text
    else:
        text = data
    ngram_count = collections.Counter(ngrams(text.lower().split(), n_gram))
    return ngram_count
