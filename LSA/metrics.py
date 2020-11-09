import numpy as np


def precision_at_k(truth: int, prediction: np.ndarray, k=10):
    return sum(prediction[:k] == truth) / k
