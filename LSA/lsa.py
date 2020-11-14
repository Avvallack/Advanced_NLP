import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

UN_SPACE_PATTERN = re.compile('')
SPACED_PATTERN = re.compile('-|\.|@|\s')
FREQUENCY_THRESHOLD = 100


def process_with_space(text):
    return re.sub(SPACED_PATTERN, ' ', text)


def basic_cleaning(word):
    return re.sub('\W', '', word)


def tokenize(text):
    text_split = process_with_space(text).lower().split()
    return [basic_cleaning(word) for word in text_split if len(basic_cleaning(word)) > 0]


def tokenizer(corpus):
    tokenized_corpus = []
    for text in corpus:
        tokenized_corpus.append(tokenize(text))
    return tokenized_corpus


def get_token_list(tokenized_corpus):
    return [token for token_list in tokenized_corpus for token in token_list]


def frequency_counts(tokenized_corpus):
    tokens_list = get_token_list(tokenized_corpus)
    return Counter(tokens_list)


def clean_corpus(tokenized_corpus, min_frequency=FREQUENCY_THRESHOLD):
    token_counts = frequency_counts(tokenized_corpus)
    tokens_to_save = [token for token, count in token_counts.items() if count > min_frequency]
    return [[token for token in token_list if token in tokens_to_save] for token_list in tokenized_corpus]


def create_dtm(tokenized_corpus):
    index_pointer = [0]
    indices = []
    data = []
    vocabulary = {}
    for doc in tokenized_corpus:
        for token in doc:
            index = vocabulary.setdefault(token, len(vocabulary))
            indices.append(index)
            data.append(1)
        index_pointer.append(len(indices))

    return csr_matrix((data, indices, index_pointer), dtype=int), vocabulary


def tf_idf_transformation(dtm):
    col_indices = dtm.nonzero()[1]
    term_occurrences = np.array(list(Counter(col_indices).values()))
    number_of_documents = dtm.shape[0]
    term_frequency = dtm.multiply(1 / dtm.sum(axis=1))
    inverse_document_frequency = np.log(number_of_documents / term_occurrences)
    return term_frequency.multiply(inverse_document_frequency).tocsr()


def lsa(vectorized_dtm, components=300, use_singulars=True):
    decomposer = TruncatedSVD(n_components=components, random_state=239)
    decomposed = decomposer.fit_transform(vectorized_dtm)
    if use_singulars:
        return decomposed.dot(np.diag(decomposer.singular_values_))
    return decomposed


def most_similar(decomposed_dtm, target, k=10):
    norm = np.sqrt((decomposed_dtm ** 2).sum(axis=1))
    normalized = np.multiply(decomposed_dtm.T, 1 / norm)
    similarities = normalized.T.dot(normalized)
    np.fill_diagonal(similarities, -1)
    target_indices = np.argpartition(-similarities, k, axis=0)[:, :k]
    most_similar_docs = target[target_indices]
    return most_similar_docs
