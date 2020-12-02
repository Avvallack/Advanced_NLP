import numpy as np
import pandas as pd
from numpy.random import normal
from random import choice

from embedding.configuration import *
from LSA.metrics import mean_precision_at_k


def initialize_embedding(vocab, dimension=300):
    return normal(0, size=(len(vocab), dimension))


def loss_function(title, tags_true, tags_wrong):
    return max(0, (1 - title.dot(tags_true) + title.dot(tags_wrong)))


def gradient(title, tags_true, tags_wrong):
    loss = loss_function(title, tags_true, tags_wrong)
    if loss == 0:
        return 0, None
    anchor_derivative = - tags_true + tags_wrong
    positive_derivative = - title
    negative_derivative = title
    return loss, [anchor_derivative, positive_derivative, negative_derivative]


def get_sum_vector(vector, vocabulary, embedding_matrix):
    vec_indices = [vocabulary[word] for word in vector]
    return embedding_matrix[vec_indices].sum(axis=0), vec_indices


def get_title_tags_similarity_matrix(index_frame: pd.DataFrame,
                                     embedding_matrix: np.ndarray,
                                     tags_col=TAGS_COL,
                                     title_col=TITLE_COL):
    title_matrix = np.zeros((index_frame.shape[0], embedding_matrix.shape[1]))
    tags_matrix = np.zeros((index_frame.shape[0], embedding_matrix.shape[1]))
    for index, row in index_frame.iterrows():
        title_matrix[index] = embedding_matrix[row[title_col]].sum(axis=0)
        tags_matrix[index] = embedding_matrix[row[tags_col]].sum(axis=0)

    return tags_matrix.dot(title_matrix.T)


def choose_triplet(clean_frame, anchor_index, title_col=TITLE_COL, tags_col=TAGS_COL):
    in_progress = True
    while in_progress:
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
             stop_rounds=5,
             title_col=TITLE_COL,
             tags_col=TAGS_COL):
    gradient_matrix = np.zeros(embedding_matrix.shape)
    best_embeddings = embedding_matrix.copy()
    best_metric = 0
    unchanged_rounds = 0
    for _ in num_iterations:
        for index in clean_frame.index:
            triplet = choose_triplet(clean_frame, index, title_col, tags_col)
            indices_list = []
            sum_vectors = []
            for vector in triplet:
                sum_vector, indices = get_sum_vector(vector, vocabulary, embedding_matrix)
                indices_list.append(indices)
                sum_vectors.append(sum_vector)

            loss, gradient_vector = gradient(*sum_vectors)
            if gradient_vector:
                for grad, idx in zip(gradient_vector, indices_list):
                    embedding_matrix[idx] -= learning_rate * grad / np.sqrt(gradient_matrix[idx] + 0.00000001)
                    gradient_matrix[idx] += np.square(grad)
        metric = mean_precision_at_k()
        if metric > best_metric:
            best_embeddings = embedding_matrix.copy()
            best_metric = metric
        else:
            unchanged_rounds += 1
        if unchanged_rounds >= stop_rounds:
            return best_embeddings
    return best_embeddings








