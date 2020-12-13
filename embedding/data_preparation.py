import re
import pandas as pd
from collections import defaultdict

from embedding.configuration import *


SPACED_PATTERN = re.compile("-|–|\\\\|:|\#|@|\.\s")
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


def build_vocab(target, train_exemplars, min_frequency=FREQUENCY_THRESHOLD):
    """
    clean only titles by frequency, cause tags in corpus already cleaned
    :param target:
    :param train_exemplars:
    :param min_frequency:
    :return: dict vocabulary of words with indices as items and words as keys
    """
    target_tokens = [token for token_list in target for token in token_list]
    train_tokens = [token for token_list in train_exemplars for token in token_list]
    tokens = target_tokens + train_tokens
    vocabulary = defaultdict(int)
    for token in tokens:
        vocabulary[token] += 1
    clean_vocabulary = defaultdict(int)
    for key, value in vocabulary.items():
        if value > min_frequency:
            clean_vocabulary[key] = len(clean_vocabulary) + 1

    return clean_vocabulary


def create_index_frame(vocab: defaultdict,
                       dataframe: pd.DataFrame,
                       train_col: str = TITLE_COL,
                       target_col: str = TAGS_COL
                       ):
    dataframe = dataframe.reset_index(drop=True)
    index_df = pd.DataFrame(columns=dataframe.columns, index=dataframe.index)

    for i, row in dataframe.iterrows():
        index_df.loc[i, target_col] = [vocab[word] for word in row[target_col]]
        index_df.loc[i, train_col] = [vocab[word] for word in row[train_col]]
    return index_df
