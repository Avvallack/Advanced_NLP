import numpy as np
import pandas as pd
from numpy.random import normal
from random import choice, choices
from tqdm.notebook import tqdm

from embedding.configuration import *


def initialize_embedding(vocabulary, dimension=300, variance=0.01):
    return normal(loc=0, scale=variance, size=(len(vocabulary) + 1, dimension)).astype(np.float32) + 0.00000001


def loss_function(train_vector, truth_target, wrong_target):
    return max(0, (1 - train_vector.dot(truth_target) + train_vector.dot(wrong_target)))


def gradient(train_vector, truth_target, wrong_target):
    loss = loss_function(train_vector, truth_target, wrong_target)
    if loss == 0:
        return 0, []
    anchor_derivative = - truth_target + wrong_target
    positive_derivative = - train_vector
    negative_derivative = train_vector
    return loss, [anchor_derivative, positive_derivative, negative_derivative]


def get_gradient_norm(gradient_vector):
    return [np.linalg.norm(element) for element in gradient_vector]


def choose_triplet(index_frame, anchor_index, train_col=TITLE_COL, target_col=TAGS_COL):
    in_progress = True
    while in_progress:
        wrong_index = choice(index_frame.index)
        if anchor_index == wrong_index:
            continue
        else:
            in_progress = False
    anchor = index_frame.iloc[anchor_index][train_col]
    positive = index_frame.iloc[anchor_index][target_col]
    negative = index_frame.iloc[wrong_index][target_col]

    return anchor, positive, negative


def get_sum_vector(vector, embedding_matrix):
    return embedding_matrix[vector].sum(axis=0), vector


def get_avg_vector(vector, embedding_matrix):
    return embedding_matrix[vector].mean(axis=0), vector


def get_title_tags_similarity_matrix(index_frame: pd.DataFrame,
                                     embedding_matrix: np.ndarray,
                                     train_col=TITLE_COL,
                                     target_col=TAGS_COL,
                                     batch_size=10000,
                                     vector_type='sum'):
    title_matrix = np.zeros((batch_size, embedding_matrix.shape[1]), dtype=np.float32)
    tags_matrix = np.zeros((batch_size, embedding_matrix.shape[1]), dtype=np.float32)
    indices = choices(index_frame.index, k=batch_size)
    for index, row in index_frame.iloc[indices].reset_index(drop=True).iterrows():
        if vector_type == 'sum':
            title_matrix[index] = embedding_matrix[row[train_col]].sum(axis=0)
            tags_matrix[index] = embedding_matrix[row[target_col]].sum(axis=0)
        elif vector_type == 'avg':
            title_matrix[index] = embedding_matrix[row[train_col]].mean(axis=0)
            tags_matrix[index] = embedding_matrix[row[target_col]].mean(axis=0)
        else:
            raise ValueError
    return title_matrix.dot(tags_matrix.T)


def get_most_similar_k(title_tags_matrix, k=10):
    return np.argpartition(-title_tags_matrix, k, axis=1)[:, :k]


def calculate_metric(most_similar_matrix):
    zeros = most_similar_matrix == np.arange(most_similar_matrix.shape[0])[:, None]
    sums = zeros.sum(axis=1)
    return sums.mean()


def gradient_vector_choice(gradient_vector, indices, vector_type='sum'):
    if vector_type == 'sum':
        return gradient_vector
    if vector_type == 'avg':
        return gradient_vector / len(indices)
    raise ValueError


def adagrad_update(gradient_vector, gradient_matrix, index_list):
    return gradient_matrix[index_list] + np.squre(gradient_vector)


def rmsprop_update(gradient_vector, gradient_matrix, index_list, update_gamma=0.95):
    return update_gamma * gradient_matrix[index_list] + (1 - update_gamma) * np.square(gradient_vector)


def train_embeddings(index_frame,
                     embedding_matrix,
                     vector_method='sum',
                     descent_type='adagrad',
                     learning_rate=0.01,
                     num_iterations=100,
                     stop_rounds=5,
                     train_col=TITLE_COL,
                     target_col=TAGS_COL,
                     update_rate=None):
    gradient_matrix = np.zeros(embedding_matrix.shape, dtype=np.float32)
    best_embeddings = embedding_matrix.copy()
    best_metric = 0
    unchanged_rounds = 0
    vector_norms = []
    if vector_method == 'sum':
        vector_function = get_sum_vector
    else:
        vector_function = get_avg_vector
    for _ in tqdm(range(num_iterations)):
        epoch_norms = []
        for index in tqdm(index_frame.index):
            triplet = choose_triplet(index_frame, index, train_col, target_col)
            indices_list = []
            sum_vectors = []
            for vector in triplet:
                sum_vector, indices = vector_function(vector, embedding_matrix)
                indices_list.append(indices)
                sum_vectors.append(sum_vector)

            loss, gradient_vector = gradient(*sum_vectors)
            epoch_norms.append(get_gradient_norm(gradient_vector))
            for grad, idx in zip(gradient_vector, indices_list):
                embedding_matrix[idx] -= (learning_rate * grad) / np.sqrt(gradient_matrix[idx] + 0.00000001)
                grad = gradient_vector_choice(grad, idx, vector_method)
                if descent_type == 'adagrad':
                    gradient_matrix[idx] = adagrad_update(grad, gradient_matrix, idx)
                elif descent_type == 'rmsprop' and update_rate:
                    gradient_matrix[idx] = rmsprop_update(grad, gradient_matrix, idx, update_rate)
                else:
                    print('wrong combination')
                    raise ValueError
        vector_norms.append(np.array([elem for elem in epoch_norms if elem]).mean(axis=0))
        doc_matrix = get_title_tags_similarity_matrix(index_frame, embedding_matrix, target_col, train_col)
        most_similar = get_most_similar_k(doc_matrix, k=10)
        metric = calculate_metric(most_similar)
        if metric > best_metric:
            best_embeddings = embedding_matrix.copy()
            best_metric = metric
            # if _ % 10 == 0:
            print("Current metric is: {:.5f}".format(best_metric))
            unchanged_rounds = 0
        else:
            unchanged_rounds += 1
            print("Current metric is: {:.5f}".format(best_metric))
        if unchanged_rounds >= stop_rounds:
            print("Current metric is: {:.3f}".format(best_metric))
            break
    return best_embeddings, vector_norms


if __name__ == '__main__':
    from embedding.data_load import DATA_FRAME
    from embedding.data_preparation import create_index_frame, build_vocab

    vocab = build_vocab(DATA_FRAME['clean_tags'], DATA_FRAME['clean_title'])
    index_df = create_index_frame(vocab, DATA_FRAME)

    initial_matrix = initialize_embedding(vocab, dimension=30, variance=0.1)
    trained_embeddings = train_embeddings(index_df,
                                          initial_matrix,
                                          vector_method='avg',
                                          descent_type='rmsprop',
                                          update_rate=0.9,
                                          stop_rounds=5,
                                          num_iterations=100)
