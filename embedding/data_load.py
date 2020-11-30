import pandas as pd
import ast
import numpy as np

from embedding.data_preparation import clean_title


PATH = 'https://dl.dropboxusercontent.com/s/5qz8lb8nbp4dk0p/title_tags.csv?dl=0'
DATA_FRAME = pd.read_csv(PATH, index_col=0)
DATA_FRAME['clean_tags'] = DATA_FRAME['clean_tags'].apply(ast.literal_eval)
DATA_FRAME['clean_title'] = DATA_FRAME['clean_title'].apply(clean_title)
DATA_FRAME['clean_tags'] = [tag if tag else np.nan for tag in DATA_FRAME.clean_tags]
DATA_FRAME.dropna(inplace=True)
DATA_FRAME.reset_index(drop=True, inplace=True)

