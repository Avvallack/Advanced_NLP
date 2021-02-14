import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dt
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict, Counter
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from collections import OrderedDict


class BiLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, embedding_size=128, num_layers=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        # Set initial states
        emb = self.embedding(x)
        lstm_out, self.hidden = self.lstm(emb)
        y_pred = self.linear(lstm_out)

        return y_pred


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
        return np.array([self.vocab[token.lower()] for token in text])

    def __replace_with_tag_index(self, tags):
        return np.array([self.tag_vocab[token] for token in tags])


class NerNN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size=256, embedding_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, num_classes)
        self.f1_metric = pl.metrics.F1(num_classes=self.num_classes, average='macro', )

    def forward(self, x):
        # Set initial states
        emb = self.embedding(x)
        lstm_out, self.hidden = self.lstm(emb)
        y_pred = self.linear(lstm_out)

        return y_pred

    def training_step(self, batch, batch_idx):
        sent, tags = batch

        outputs = self.forward(sent)
        loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data_batch, batch_nb):
        sen, tags = data_batch
        outputs = self.forward(sen)
        loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        tag_mask = tags != 0
        predicted = outputs.argmax(2)
        f1_score = self.f1_metric(predicted[tag_mask], tags[tag_mask])
        self.log('val_loss', loss_val)
        self.log('val_macro_f1', f1_score)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_f1': f1_score,
        })
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def padding(data):
    x, y = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x, y


def padded_data_loader(data, workers, batch_size=32):
    return dt.DataLoader(dataset=data, batch_size=batch_size, collate_fn=padding, num_workers=workers)
