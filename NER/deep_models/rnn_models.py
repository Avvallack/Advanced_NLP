import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from collections import OrderedDict
from argparse import ArgumentParser
from torchcrf import CRF


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

    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.num_classes = num_classes
        assert self.hparams.rnn_type in ['lstm', 'qrnn', 'cnn'], 'RNN type is not supported'

        self.embedding = nn.Embedding(self.input_size, self.hparams.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(self.hparams.dropout)
        if self.hparams.rnn_type == 'lstm':
            self.lstm = nn.LSTM(self.hparams.embedding_size,
                                self.hparams.hidden_size,
                                self.hparams.num_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=self.hparams.dropout)
        elif self.hparams.rnn_type == 'cnn':
            self.word2cnn = nn.Linear(self.hparams.embedding_size, self.hparams.hidden_size * 2)
            self.cnn_list = list()
            for _ in range(self.num_layers):
                self.cnn_list.append(nn.Conv1d(self.hparams.hidden_size * 2,
                                               self.hparams.hidden_size * 2,
                                               kernel_size=3,
                                               padding=1, ))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(self.hparams.dropout))
                self.cnn_list.append(nn.BatchNorm1d(self.hparams.hidden_size * 2))
            self.cnn = nn.Sequential(*self.cnn_list)
        elif self.hparams.rnn_type == 'qrnn':
            raise NotImplementedError("Isn't implemented yet")
        if self.hparams.use_crf:
            self.crf = CRF(num_tags=self.num_classes, batch_first=True)
        self.linear = nn.Linear(self.hparams.hidden_size * 2, self.num_classes)

        self.f1_metric = tm.F1Score(num_classes=self.num_classes, average='macro', )

    def forward(self, x):
        # Set initial states
        emb = self.embedding(x)
        emb = self.dropout(emb)
        if self.hparams.rnn_type == 'lstm':
            output, self.hidden = self.lstm(emb)
        elif self.hparams.rnn_type == 'cnn':
            to_cnn = torch.tanh(self.word2cnn(emb)).transpose(2, 1).contiguous()
            output = self.cnn(to_cnn).transpose(1, 2).contiguous()
        y_pred = self.linear(output)

        return y_pred

    def training_step(self, batch, batch_idx):
        sent, tags = batch

        outputs = self.forward(sent)
        if self.hparams.use_crf:
            mask_sent = sent.eq(0).eq(0)
            loss = -self.crf.forward(outputs, tags, mask_sent)
        else:
            loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data_batch, batch_nb):
        sent, tags = data_batch
        outputs = self.forward(sent)
        if self.hparams.use_crf:
            mask_sent = sent.eq(0).eq(0).type(torch.ByteTensor)
            loss_val = -self.crf.forward(outputs, tags, mask_sent)
            predicted = torch.tensor(self.crf.decode(outputs))
        else:
            loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
            predicted = outputs.argmax(2)

        mask = tags.eq(0).eq(0)
        f1_score = self.f1_metric(predicted[mask], tags[mask])

        output = OrderedDict({
            'val_loss': loss_val,
            'val_f1': f1_score,
        })
        self.log_dict(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                                      eps=self.hparams.adam_eps,
                                      weight_decay=self.hparams.weight_decay,
                                      )
        return optimizer
