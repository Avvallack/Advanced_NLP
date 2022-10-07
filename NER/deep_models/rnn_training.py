import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix

from NER.data_load import get_ner_dataset
from ner_dataset import NerDataSet, padded_data_loader
from rnn_models import NerRNN


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser = NerRNN.add_model_specific_args(parser)
    args = parser.parse_args()

    sent, tags = get_ner_dataset()
    ds = NerDataSet(sent, tags)
    train_set, test_set = random_split(ds, [40000, 7959], generator=torch.Generator().manual_seed(42))
    train_loader = padded_data_loader(data=train_set, batch_size=args.batch_size, workers=0)
    test_loader = padded_data_loader(data=test_set, batch_size=args.batch_size, workers=0)

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
    print("f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(report.f1.mean(), report.recall.mean(),
                                                                       report.precision.mean()))
