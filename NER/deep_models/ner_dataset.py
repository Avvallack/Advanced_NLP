import numpy as np
import torch
import torch.utils.data as dt
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter


class NerDataSet(dt.Dataset):
    def __init__(self, sentences, tags):
        self.__build_vocab(sentences)
        self.__build_tags_vocab(tags)
        self.x = [self.__replace_with_index(text) for text in sentences]
        self.y = [self.__replace_with_tag_index(tag) for tag in tags]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    def __len__(self):
        return len(self.x)

    def __build_vocab(self, texts):
        words = [word.lower() for words in texts for word in words]
        cnt = Counter(words).most_common()
        self.vocab = defaultdict(int)
        self.vocab['PAD'] = 0
        for i, (token, _) in enumerate(cnt):
            self.vocab[token] = i + 1
        self.vocab['UNK'] = len(self.vocab)

    def __build_tags_vocab(self, tags):
        tags = list(set([tag for tag_el in tags for tag in tag_el]))
        self.tag_vocab = defaultdict(int)
        self.tag_vocab['PAD'] = 0
        for token in tags:
            self.tag_vocab[token] = len(self.tag_vocab)

    def __replace_with_index(self, text):
        return np.array([self.vocab[token.lower()] for token in text], dtype='int64')

    def __replace_with_tag_index(self, tags):
        return np.array([self.tag_vocab[token] for token in tags], dtype='int64')


def padding(data):
    x, y = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x, y


def padded_data_loader(data, workers, batch_size=32):
    return dt.DataLoader(dataset=data, batch_size=batch_size, collate_fn=padding, num_workers=workers)
