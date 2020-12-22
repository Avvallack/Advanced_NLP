import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from random import choice

from embedding.configuration import *


def initialize_embedding(vocabulary, dimension=300):
    return np.random.randn(len(vocabulary), dimension).astype(np.float32) * 1e-3


def loss_function(train_vector, truth_target, wrong_target):
    return max(0, (1 - train_vector.dot(truth_target) + train_vector.dot(wrong_target)))


def gradient(anchor, truth_target, wrong_target):
    loss = loss_function(anchor, truth_target, wrong_target)
    if loss == 0:
        return 0, []
    anchor_derivative = - truth_target + wrong_target
    positive_derivative = - anchor
    negative_derivative = anchor
    return loss, [anchor_derivative, positive_derivative, negative_derivative]


def get_gradient_norm(gradient_vector):
    return [np.linalg.norm(element) for element in gradient_vector]


def choose_triplet(index_frame, anchor_index, wrong_index,
                   train_col=TITLE_COL, target_col=TAGS_COL, delete_common=False):
    anchor = index_frame.iloc[anchor_index][train_col]
    positive = index_frame.iloc[anchor_index][target_col]
    negative = index_frame.iloc[wrong_index][target_col]
    if delete_common:
        intersection = set(anchor).intersection(positive)
        if intersection:
            col = choice(('anchor', 'positive'))
            if col == 'anchor':
                if len(anchor) > len(intersection):
                    anchor = list(set(anchor) - intersection)
                elif len(positive) > len(intersection):
                    positive = list(set(positive) - intersection)

            else:
                if len(positive) > len(intersection):
                    positive = list(set(positive) - intersection)
                elif len(anchor) > len(intersection):
                    anchor = list(set(anchor) - intersection)

    return anchor, positive, negative


def get_sum_vector(vector, embedding_matrix):
    return embedding_matrix[vector].sum(axis=0), vector


def get_avg_vector(vector, embedding_matrix):
    return embedding_matrix[vector].mean(axis=0), vector


def get_documents_matrix(dataframe_column, embedding_matrix, vector_type):
    n_row = len(dataframe_column)
    dataframe_column = dataframe_column.reset_index(drop=True).explode()
    sparse_col = csr_matrix(
        (np.ones(len(dataframe_column)),
         (dataframe_column.index.values,
          dataframe_column.values)),
        shape=(n_row, embedding_matrix.shape[0])
    )
    embed_vectors = sparse_col.dot(embedding_matrix)
    if vector_type == 'sum':
        return embed_vectors
    return embed_vectors / sparse_col.sum(axis=1)


def get_title_tags_similarity_matrix(index_frame: pd.DataFrame,
                                     embedding_matrix: np.ndarray,
                                     train_col=TITLE_COL,
                                     target_col=TAGS_COL,
                                     test_size=10000,
                                     vector_type='sum'):
    train_docs = get_documents_matrix(index_frame[train_col].tail(test_size), embedding_matrix, vector_type)
    target_docs = get_documents_matrix(index_frame[target_col].tail(test_size), embedding_matrix, vector_type)
    similarity = train_docs.dot(target_docs.T)
    return similarity


def get_most_similar_k(title_tags_matrix, k=10):
    return np.argpartition(-title_tags_matrix, k)[:k, :]


def calculate_metric(most_similar_matrix):
    zeros = (most_similar_matrix == np.arange(most_similar_matrix.shape[1])[None, :]).mean()
    return zeros


def gradient_vector_choice(gradient_vector, indices, vector_type='sum'):
    if vector_type == 'sum':
        return gradient_vector
    if vector_type == 'avg':
        return gradient_vector / len(indices)
    raise ValueError('Incorrect vector method was passed')


def adagrad_update(gradient_vector, gradient_matrix, index_list):
    return gradient_matrix[index_list] + np.square(gradient_vector)


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
                     update_rate=None,
                     test_size=5000,
                     k=5,
                     delete_common=False):
    gradient_matrix = np.full_like(embedding_matrix, 1e-8,  dtype=np.float32)
    best_embeddings = embedding_matrix.copy()
    best_metric = 0
    unchanged_rounds = 0
    if vector_method == 'sum':
        vector_function = get_sum_vector
    elif vector_method == 'avg':
        vector_function = get_avg_vector
    else:
        raise ValueError('Incorrect vector method was passed')
    for _ in tqdm(range(num_iterations)):
        true_idx = np.random.permutation(index_frame.shape[0] - test_size)
        false_idx = np.roll(true_idx, 1)
        for true_index, false_index in zip(true_idx, false_idx):
            triplet = choose_triplet(index_frame, true_index, false_index, train_col, target_col, delete_common)
            indices_list = []
            sum_vectors = []
            for vector in triplet:
                sum_vector, indices = vector_function(vector, embedding_matrix)
                indices_list.append(indices)
                sum_vectors.append(sum_vector)

            loss, gradient_vector = gradient(*sum_vectors)
            for grad, idx in zip(gradient_vector, indices_list):
                grad = gradient_vector_choice(grad, idx, vector_method)
                if descent_type == 'adagrad':
                    gradient_matrix[idx] = adagrad_update(grad, gradient_matrix, idx)
                elif descent_type == 'rmsprop' and update_rate:
                    gradient_matrix[idx] = rmsprop_update(grad, gradient_matrix, idx, update_rate)
                else:
                    raise ValueError('Incorrect optimization method was passed')

                embedding_matrix[idx] -= (learning_rate * grad) / np.sqrt(gradient_matrix[idx])

        doc_matrix = get_title_tags_similarity_matrix(index_frame,
                                                      embedding_matrix,
                                                      target_col,
                                                      train_col,
                                                      test_size,
                                                      vector_method)
        most_similar = get_most_similar_k(doc_matrix, k=k)
        metric = calculate_metric(most_similar)
        print(f"epoch = {_} R@{k} = {100 * metric:.2f}%")
        if metric > best_metric:
            best_embeddings = embedding_matrix.copy()
            best_metric = metric
            unchanged_rounds = 0
        else:
            unchanged_rounds += 1
        if unchanged_rounds >= stop_rounds:
            print("Best metric is: {:.2f}%".format(best_metric * 100))
            break
    return best_embeddings


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
