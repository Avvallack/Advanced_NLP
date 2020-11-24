import numpy as np
from numpy.random import normal


def initialize_embedding(vocab, dimension=300):
    return normal(0, size=(dimension, len(vocab)))


def loss_function(title, tags_true, tags_wrong):
    return max(0, (1 - title.dot(tags_true) + title.dot(tags_wrong)))


def gradient(title, tags_true, tags_wrong):
    if loss_function(title, tags_true, tags_wrong) == 0:
        return 0
    anchor = 2 * (tags_true - tags_wrong)
    positive = -2 * (title - tags_true)
    negative = 2 * (title - tags_wrong)
    return (anchor, positive, negative)



def adagrad(**args):
    pass
