import numpy as np



def initialize_embedding(corpus, dimension=300):
    pass


def loss_function(title, tags_true, tags_wrong):
    return max(0, (1 - title.dot(tags_true) + title.dot(tags_wrong)))


def gradient(loss_function):
    pass


def adagrad(**args):
    pass
