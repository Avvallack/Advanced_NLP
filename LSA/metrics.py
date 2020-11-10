import numpy as np


def mean_precision_at_k(truth: np.ndarray, prediction: np.ndarray, k=10):
    prediction_slice = prediction[:, :k]
    zero_similar = np.add(prediction_slice.T, -truth)
    return np.mean(np.count_nonzero(zero_similar == 0, axis=0) / k)
