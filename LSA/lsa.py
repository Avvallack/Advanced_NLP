import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


UN_SPACE_PATTERN = re.compile('')
SPACED_PATTERN = re.compile('-|\.|\@|\s')
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


def get_dictionary(tokenized_corpus):
    tokens_list = get_token_list(tokenized_corpus)
    return list(set(tokens_list))


def create_dtm(tokenized_corpus):
    dictionary = get_dictionary(tokenized_corpus)
    bow_corpus = []
    for token_list in tokenized_corpus:
        bow = [0] * len(dictionary)
        for token in token_list:
            bow[dictionary.index(token)] += 1
        bow_corpus.append(bow)
    return bow_corpus, dictionary


def vectorize_dtm(dtm):
    words_per_doc = np.sum(dtm, axis=1)
    word_frequency = np.count_nonzero(dtm, axis=0)
    number_of_documents = len(dtm)
    term_frequency = np.array(dtm).T / words_per_doc
    inverse_document_frequency = np.log(number_of_documents / word_frequency)
    return term_frequency.T * inverse_document_frequency


def lsa(vectorized_dtm, components=300):
    decompositor = TruncatedSVD(n_components=components)
    decomposed = decompositor.fit_transform(vectorized_dtm)
    return decomposed, decompositor.components_


def most_similar(decomposed_dtm, targets, k=10):
    results = []
    for i, document in enumerate(decomposed_dtm):
        similarity = cosine_similarity(decomposed_dtm, document.reshape(1, -1))
        most_similar_indexes = similarity.reshape(1, -1)[0].argsort()[-k + 1:-1][::-1]
        results.append({'truth': targets[i], 'similar_docs': targets[most_similar_indexes]})
    return pd.DataFrame.from_records(results)
