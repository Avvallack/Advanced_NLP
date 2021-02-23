import pandas as pd


def get_ner_dataset():
    df = pd.read_csv('https://dl.dropboxusercontent.com/s/tlijezgr8tnpeym/ner_dataset.csv?dl=0',
                     header=0,
                     encoding='latin')

    df['Sentence #'].fillna(method='ffill', inplace=True)
    grouped = df.groupby(by='Sentence #').agg(lambda x: list(x))

    sentences = grouped['Word'].values
    tags = grouped['Tag'].values
    return sentences, tags
