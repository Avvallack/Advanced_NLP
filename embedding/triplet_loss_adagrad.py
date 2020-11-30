import numpy as np
from numpy.random import normal
from random import choice

from embedding.configuration import *


def initialize_embedding(vocab, dimension=300):
    return normal(0, size=(len(vocab), dimension))


def loss_function(title, tags_true, tags_wrong):
    return max(0, (1 - title.dot(tags_true) + title.dot(tags_wrong)))


def gradient(title, tags_true, tags_wrong):
    if loss_function(title, tags_true, tags_wrong) == 0:
        return 0
    anchor_derivative = - tags_true + tags_wrong
    positive_derivative = - title
    negative_derivative = title
    return anchor_derivative, positive_derivative, negative_derivative


def get_sum_vector(vector, vocabulary, embedding_matrix):
    vec_indices = [vocabulary[word] for word in vector]
    return embedding_matrix[vec_indices].sum(axis=0)


def choose_triplet(clean_frame, title_col=TITLE_COL, tags_col=TAGS_COL):
    in_progress = True
    while in_progress:
        anchor_index = choice(clean_frame.index)
        wrong_index = choice(clean_frame.index)
        if anchor_index == wrong_index:
            continue
        else:
            in_progress = False
    anchor = clean_frame.iloc[anchor_index][title_col]
    positive = clean_frame.iloc[anchor_index][tags_col]
    negative = clean_frame.iloc[wrong_index][tags_col]

    return anchor, positive, negative


def ada_grad(clean_frame,
             embedding_matrix,
             vocabulary,
             learning_rate=0.01,
             num_iterations=10000,
             title_col=TITLE_COL,
             tags_col=TAGS_COL):
    gradient_matrix = np.zeros((embedding_matrix.shape[1], embedding_matrix.shape[1]))
    for i in range(num_iterations):
        anchor, positive, negative = choose_triplet(clean_frame, title_col, tags_col)
        vec_anchor = get_sum_vector(anchor, vocabulary, embedding_matrix)
        vec_positive = get_sum_vector(positive, vocabulary, embedding_matrix)
        vec_negative = get_sum_vector(negative, vocabulary, embedding_matrix)
        loss = loss_function(vec_anchor, positive, negative)
        pass



