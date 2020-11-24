import re

from LSA.lsa import clean_corpus

SPACED_PATTERN = re.compile("-|–|\\\\|:|\#|@")
FREQUENCY_THRESHOLD = 100


def process_sharp(tag):
    return re.sub('c\#|с\#', 'c_sharp', tag)


def process_f_sharp(tag):
    return re.sub('f\#', 'f_sharp', tag)


def process_plus(tag):
    return re.sub('c\+\+|с\+\+|си\+\+', 'c_plus_plus', tag)


def process_dot(tag):
    return re.sub('\.', '_dot_', tag)


def process_slash(tag):
    return re.sub("/|\s/\s", '_', tag)


def process_with_space(tag):
    return re.sub(SPACED_PATTERN, ' ', tag)


def process_without_space(text):
    text = re.sub('\s', '_', text)
    text = re.sub('\W', '', text)
    return re.sub('_', ' ', text)


def clean_title(text):
    text = process_dot(text)
    text = process_plus(text)
    text = process_sharp(text)
    text = process_f_sharp(text)
    text = process_slash(text)
    text = process_with_space(text)
    return process_without_space(text).split()


def build_vocab(tags, tokenized_titles, min_frequency=FREQUENCY_THRESHOLD):
    """
    clean only titles by frequency, cause tags in corpus already cleaned
    :param tags:
    :param tokenized_titles:
    :param min_frequency:
    :return: dict vocabulary of words with indices as items and words as keys
    """
    tags = set([tag for tag_list in tags for tag in tag_list])
    titles = clean_corpus(tokenized_titles, min_frequency)
    title_tokens = set([token for token_list in titles for token in token_list])
    token_list = list(tags.union(title_tokens))
    vocabulary = {token: token_list.index(token) for token in token_list}
    return vocabulary

