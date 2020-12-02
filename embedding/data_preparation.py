import re
import pandas as pd

from embedding.configuration import *
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


def finalize_dataframe(vocab, dataframe, col_tags=TAGS_COL, col_title=TITLE_COL):
    dataframe = dataframe.copy()
    dataframe[col_tags] = [[tag for tag in tags if tag in vocab.keys()]
                           for tags in dataframe[col_tags]]
    dataframe[col_title] = [[token for token in title if token in vocab.keys()]
                            for title in dataframe[col_title]]
    return dataframe


def create_index_frame(vocab: dict,
                       finalized_df: pd.DataFrame,
                       col_tags: str = TAGS_COL,
                       col_title: str = TITLE_COL):
    index_df = pd.DataFrame(columns=finalized_df.columns, index=finalized_df.index)
    for i, row in finalized_df.iterrows():
        index_df.loc[i, col_tags] = [vocab[word] for word in row[col_tags]]
        index_df.loc[i, col_title] = [vocab[word] for word in row[col_title]]
    return index_df
