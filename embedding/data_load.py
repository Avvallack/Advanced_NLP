import pandas as pd
import ast
import numpy as np

from embedding.data_preparation import clean_title


PATH = 'https://dl.dropboxusercontent.com/s/5qz8lb8nbp4dk0p/title_tags.csv?dl=0'
DATA_FRAME = pd.read_csv(PATH, index_col=0)
DATA_FRAME['clean_tags'] = DATA_FRAME['clean_tags'].apply(ast.literal_eval)
DATA_FRAME['clean_title'] = DATA_FRAME['clean_title'].apply(clean_title)
DATA_FRAME['clean_tags'] = [tag if tag else np.nan for tag in DATA_FRAME.clean_tags]
DATA_FRAME['clean_title'] = [token if token else np.nan for token in DATA_FRAME.clean_title]
DATA_FRAME.dropna(inplace=True)
DATA_FRAME.reset_index(drop=True, inplace=True)

TEXT_PATH = 'https://dl.dropboxusercontent.com/s/glz13exvbzprtw3/titles_texts_habr.csv?dl=0'
TEXT_FRAME = pd.read_csv(TEXT_PATH, index_col=0)
TEXT_FRAME['clean_title'] = TEXT_FRAME['clean_title'].apply(ast.literal_eval)
TEXT_FRAME['clean_text'] = TEXT_FRAME['clean_text'].apply(ast.literal_eval)
TEXT_FRAME['clean_text'] = [token if token else np.nan for token in TEXT_FRAME.clean_text]
TEXT_FRAME['clean_title'] = [token if token else np.nan for token in TEXT_FRAME.clean_title]
TEXT_FRAME.dropna(inplace=True)
TEXT_FRAME.reset_index(drop=True, inplace=True)

SIMPLE_TITLE = DATA_FRAME['clean_title'].apply(lambda x: x[0])
