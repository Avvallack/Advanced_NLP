import torch
import copy
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn
from torch.nn import functional as F
from attention_layer import MultiHeadAttention, FeedForward, Normalization


def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.norm_1 = Normalization(model_dim)
        self.norm_2 = Normalization(model_dim)

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FeedForward(model_dim, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_input = self.norm_1(x)
        attn = self.attention(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout_1(attn)
        norm_x = self.norm_2(x)
        feed_forwarded = self.feed_forward(norm_x)
        x = x + self.dropout_2(feed_forwarded)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.norm_1 = Normalization(model_dim)
        self.norm_2 = Normalization(model_dim)
        self.norm_3 = Normalization(model_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FeedForward(model_dim, dropout=dropout)

    def forward(self, x, enc_output, mask=None):
        attn_input = self.norm_1(x)
        attn_self = self.attn_1(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout_1(attn_self)
        norm_x = self.norm_2(x)
        attn_enc = self.attn_2(norm_x, enc_output, enc_output, mask)
        x = x + self.dropout_2(attn_enc)
        norm_out = self.norm_3(x)
        feed_forwarded = self.feed_forward(norm_out)
        x = x + self.dropout_3(feed_forwarded)
        return x


class NERTransformer(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, **kwargs):
        super(NERTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.model_dim = kwargs['model_dim']
        self.num_heads = kwargs['num_heads']
        self.num_layers = kwargs['num_layers']
        self.dropout = kwargs['dropout']
        self.embedding = nn.Embedding(vocab_size, self.model_dim)
        self.layers = get_clones(EncoderLayer(self.model_dim, self.num_heads, self.dropout),
                                 self.num_layers)
        self.norm = Normalization(self.model_dim)
        self.linear = nn.Linear(self.model_dim, num_classes)
        self.f1_metric = pl.metrics.F1(num_classes=self.num_classes, average='macro', )

    def forward(self, x, mask=None):
        embedding = self.embedding(x)

        for i in range(self.num_layers):
            embedding = self.layers[i](embedding, mask)

        norm_embedding = self.norm(embedding)
        output = self.linear(norm_embedding)
        return output

    def training_step(self, batch, batch_idx):
        sent, tags = batch
        outputs = self.forward(sent)
        loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sent, tags = batch
        outputs = self.forward(sent)
        loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=0)

        mask = tags.eq(0).eq(0)
        predicted = outputs.argmax(2)
        f1_score = self.f1_metric(predicted[mask], tags[mask])

        output = OrderedDict({
            'val_loss': loss_val,
            'val_f1': f1_score,
        })
        self.log_dict(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--model_dim", type=int, default=512)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--learning_rate", type=float, default=5e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.95)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--dropout", type=float, default=0.1)

        return parser


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
    parser = NERTransformer.add_model_specific_args(parser)

    args = parser.parse_args()

    sent, tags = get_ner_dataset()
    ds = NerDataSet(sent, tags)
    train_set, test_set = random_split(ds, [40000, 7959], generator=torch.Generator().manual_seed(42))
    train_loader = padded_data_loader(data=train_set, batch_size=64, workers=0)
    test_loader = padded_data_loader(data=test_set, batch_size=64, workers=0)

    model = NERTransformer(len(ds.vocab), len(ds.tag_vocab), **vars(args))
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
    print("f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(report.f1.mean(), report.recall.mean(),
                                                                       report.precision.mean()))
