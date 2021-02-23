import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser

from crf_layer import CRF


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


class NerRNN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=5e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.95)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--rnn_type", type=str, default='lstm')
        parser.add_argument("--use_crf", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--embedding_size", type=int, default=128)
        return parser

    def save_model_params(self, input_size, num_classes, **vars):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = vars['num_layers']
        self.rnn_type = vars['rnn_type']
        self.hidden_size = vars['hidden_size']
        self.embedding_size = vars['embedding_size']
        self.drop = vars['dropout']
        self.use_crf = vars['use_crf']

    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__()
        self.save_model_params(input_size, num_classes, **kwargs)
        assert self.rnn_type in ['lstm', 'qrnn', 'cnn'], 'RNN type is not supported'

        self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(self.drop)
        if self.rnn_type == 'lstm':
            self.lstm = nn.LSTM(self.embedding_size,
                                self.hidden_size,
                                self.num_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=self.drop)
        elif self.rnn_type == 'cnn':
            self.word2cnn = nn.Linear(self.embedding_size, self.hidden_size * 2)
            self.cnn_list = list()
            for _ in range(self.num_layers):
                self.cnn_list.append(nn.Conv1d(self.hidden_size * 2, self.hidden_size * 2, kernel_size=3, padding=1, ))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(self.drop))
                self.cnn_list.append(nn.BatchNorm1d(self.hidden_size * 2))
            self.cnn = nn.Sequential(*self.cnn_list)
        elif self.rnn_type == 'qrnn':
            raise NotImplementedError("Isn't implemented yet")
        if self.use_crf:
            self.linear = nn.Linear(self.hidden_size * 2, self.num_classes + 2)
            self.crf = CRF(self.num_classes)
        else:
            self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)

        self.f1_metric = pl.metrics.F1(num_classes=self.num_classes, average='macro', )

    def forward(self, x):
        # Set initial states
        emb = self.embedding(x)
        emb = self.dropout(emb)
        if self.rnn_type == 'lstm':
            output, self.hidden = self.lstm(emb)
        elif self.rnn_type == 'cnn':
            to_cnn = torch.tanh(self.word2cnn(emb)).transpose(2, 1).contiguous()
            output = self.cnn(to_cnn).transpose(1, 2).contiguous()
        if self.use_crf:
            x_mask = x.eq(0).eq(0)
            scores, y_pred = self.crf._viterbi_decode(output, x_mask)
        else:
            y_pred = self.linear(output)

        return y_pred

    def training_step(self, batch, batch_idx):
        sent, tags = batch

        outputs = self.forward(sent)
        if self.use_crf:
            mask_sent = sent.eq(0).eq(0)
            loss = self.crf.neg_log_likelihood_loss(outputs, mask_sent, tags)
        else:
            loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data_batch, batch_nb):
        sent, tags = data_batch
        outputs = self.forward(sent)
        loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        mask = tags.eq(0).eq(0)
        predicted = outputs.argmax(2)
        f1_score = self.f1_metric(predicted[mask], tags[mask])
        self.log('val_loss', loss_val)
        self.log('val_macro_f1', f1_score)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_f1': f1_score,
        })
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from torch.utils.data import random_split
    from sklearn.metrics import confusion_matrix

    from NER.data_load import get_ner_dataset
    from ner_dataset import NerDataSet, padded_data_loader


    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser = NerRNN.add_model_specific_args(parser)

    args = parser.parse_args()

    sent, tags = get_ner_dataset()
    ds = NerDataSet(sent, tags)
    train_set, test_set = random_split(ds, [40000, 7959], generator=torch.Generator().manual_seed(42))
    train_loader = padded_data_loader(data=train_set, batch_size=64, workers=0)
    test_loader = padded_data_loader(data=test_set, batch_size=64, workers=0)

    model = NerRNN(len(ds.vocab), len(ds.tag_vocab), **vars(args))
    logger = pl.loggers.TensorBoardLogger(args.log_dir, name=args.run_name)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, train_loader, test_loader)

    labels = list(range(1, 18))
    model.eval()
    conf_matrix = np.zeros((len(ds.tag_vocab) - 1, len(ds.tag_vocab) - 1))
    for sentences, tags in test_loader:
        tags = tags.flatten()
        tag_mask = tags != 0
        outputs = model(sentences)
        predicted = outputs.argmax(2)
        tags = tags[tag_mask]
        predicted = predicted.flatten()[tag_mask]
        conf_matrix += confusion_matrix(tags.numpy(), predicted.numpy(), labels=labels)
    tp = np.diagonal(conf_matrix)
    prec = tp / conf_matrix.sum(axis=0)
    rec = tp / conf_matrix.sum(axis=1)
    mask = np.logical_and(prec == 0, rec == 0)
    f1 = 2 * (prec * rec / (prec + rec))
    f1[mask] = 0

    labels = list(ds.tag_vocab.keys())[1:]
    report = pd.DataFrame.from_dict({'labels': labels, 'recall': rec, 'precision': prec, 'f1': f1})
    report = report.set_index('labels')
    report.fillna(0, inplace=True)
    print(report)
    print("f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(report.f1.mean(), report.recall.mean(), report.precision.mean()))
