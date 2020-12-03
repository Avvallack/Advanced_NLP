import numpy as np
import pandas as pd
from numpy.random import normal
from random import choice, choices
from tqdm.notebook import tqdm

from embedding.configuration import *


def initialize_embedding(vocab, dimension=300):
    return normal(loc=0, scale=0.5, size=(len(vocab), dimension)).astype('Float32')


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


def choose_triplet(index_frame, anchor_index, title_col=TITLE_COL, tags_col=TAGS_COL):
    in_progress = True
    while in_progress:
        wrong_index = choice(index_frame.index)
        if anchor_index == wrong_index:
            continue
        else:
            in_progress = False
    anchor = index_frame.iloc[anchor_index][title_col]
    positive = index_frame.iloc[anchor_index][tags_col]
    negative = index_frame.iloc[wrong_index][tags_col]

    return anchor, positive, negative


def get_sum_vector(vector, embedding_matrix):
    return embedding_matrix[vector].sum(axis=0), vector


def get_title_tags_similarity_matrix(index_frame: pd.DataFrame,
                                     embedding_matrix: np.ndarray,
                                     tags_col=TAGS_COL,
                                     title_col=TITLE_COL,
                                     batch_size=10000):
    title_matrix = np.zeros((batch_size, embedding_matrix.shape[1]))
    tags_matrix = np.zeros((batch_size, embedding_matrix.shape[1]))
    indices = choices(index_frame.index, k=batch_size)
    for index, row in index_frame.iloc[indices].reset_index(drop=True).iterrows():
        title_matrix[index] = embedding_matrix[row[title_col]].sum(axis=0)
        tags_matrix[index] = embedding_matrix[row[tags_col]].sum(axis=0)
    return title_matrix.dot(tags_matrix.T)


def get_most_similar_k(title_tags_matrix, k=10):
    return np.argpartition(np.multiply(-1, title_tags_matrix), k, axis=1)[:, :10]


def calculate_metric(most_similar_matrix):
    zeros = np.add(most_similar_matrix.T, np.multiply(-1, np.arange(most_similar_matrix.shape[0])))
    return np.sum(np.count_nonzero(zeros == 0, axis=0)) / zeros.shape[1]


def ada_grad(index_frame,
             embedding_matrix,
             learning_rate=0.01,
             num_iterations=1000,
             stop_rounds=10,
             tags_col=TAGS_COL,
             title_col=TITLE_COL):
    gradient_matrix = np.zeros(embedding_matrix.shape, dtype='Float64')
    best_embeddings = embedding_matrix.copy()
    best_metric = 0
    unchanged_rounds = 0
    for _ in tqdm(range(num_iterations)):
        for index in tqdm(index_frame.index):
            triplet = choose_triplet(index_frame, index, title_col, tags_col)
            indices_list = []
            sum_vectors = []
            for vector in triplet:
                sum_vector, indices = get_sum_vector(vector, embedding_matrix)
                indices_list.append(indices)
                sum_vectors.append(sum_vector)

            loss, gradient_vector = gradient(*sum_vectors)
            if gradient_vector:
                for grad, idx in zip(gradient_vector, indices_list):
                    embedding_matrix[idx] -= np.multiply(np.multiply(learning_rate, grad), 1 / np.sqrt(gradient_matrix[idx] + 0.00000001))
                    gradient_matrix[idx] += np.square(grad)
        doc_matrix = get_title_tags_similarity_matrix(index_frame, embedding_matrix, tags_col, title_col)
        most_similar = get_most_similar_k(doc_matrix, k=10)
        metric = calculate_metric(most_similar)
        if metric > best_metric:
            best_embeddings = embedding_matrix.copy()
            best_metric = metric
            if _ % 10 == 0:
                print("Current metric is: {:.3f}".format(best_metric))
        else:
            unchanged_rounds += 1
        if unchanged_rounds >= stop_rounds:
            print("Current metric is: {:.3f}".format(best_metric))
            return best_embeddings
    return best_embeddings








