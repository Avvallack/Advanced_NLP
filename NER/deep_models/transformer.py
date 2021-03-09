import torch
import copy
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn
from torch.nn import functional as F
from attention_layer import MultiHeadAttention, FeedForward, PositionalEncoding


def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FeedForward(model_dim, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_input = self.norm_1(x)
        attn = self.attention(attn_input)
        x = x + self.dropout_1(attn)
        norm_x = self.norm_2(x)
        feed_forwarded = self.feed_forward(norm_x)
        x = x + self.dropout_2(feed_forwarded)
        return x


class NERTransformer(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, **kwargs):
        super(NERTransformer, self).__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.pos_enc = PositionalEncoding(self.hparams.model_dim, self.hparams.dropout)
        self.layers = get_clones(EncoderLayer(self.hparams.model_dim, self.hparams.num_heads, self.hparams.dropout),
                                 self.hparams.num_layers)
        self.norm = nn.LayerNorm(self.hparams.model_dim)
        self.linear = nn.Linear(self.hparams.model_dim, num_classes)
        self.f1_metric = pl.metrics.F1(num_classes=self.num_classes, average='macro', )

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = self.pos_enc(embedding)
        for i in range(self.num_layers):
            embedding = self.layers[i](embedding)

        norm_embedding = self.norm(embedding)
        output = self.linear(norm_embedding)
        return output

    def training_step(self, batch, batch_idx, pad_index=0):
        sent, tags = batch
        outputs = self.forward(sent)
        loss = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=pad_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, pad_index=0):
        sent, tags = batch
        outputs = self.forward(sent)
        loss_val = F.cross_entropy(torch.flatten(outputs, 0, 1), torch.flatten(tags, 0, 1), ignore_index=pad_index)
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
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                                      eps=self.hparams.adam_eps,
                                      weight_decay=self.hparams.weight_decay,
                                      )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--model_dim", type=int, default=512)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.95)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--dropout", type=float, default=0.1)

        return parser
