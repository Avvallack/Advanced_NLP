import torch
import copy
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn
from torch.nn import functional as F
from deep_models.attention_layer import MultiHeadAttention, FeedForward, PositionalEncoding


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
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.model_dim = kwargs['model_dim']
        self.num_heads = kwargs['num_heads']
        self.num_layers = kwargs['num_layers']
        self.dropout = kwargs['dropout']
        self.embedding = nn.Embedding(vocab_size, self.model_dim)
        self.pos_enc = PositionalEncoding(self.model_dim, self.dropout)
        self.layers = get_clones(EncoderLayer(self.model_dim, self.num_heads, self.dropout),
                                 self.num_layers)
        self.norm = nn.LayerNorm(self.model_dim)
        self.linear = nn.Linear(self.model_dim, num_classes)
        self.f1_metric = pl.metrics.F1(num_classes=self.num_classes, average='macro', )

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = self.pos_enc(embedding)
        for i in range(self.num_layers):
            embedding = self.layers[i](embedding)

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
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.95)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--dropout", type=float, default=0.1)

        return parser
