import pandas as pd
import ast


PATH = 'https://dl.dropboxusercontent.com/s/5qz8lb8nbp4dk0p/title_tags.csv?dl=0'
DATA_FRAME = pd.read_csv(PATH)
DATA_FRAME['clean_tags'] = DATA_FRAME['clean_tags'].apply(ast.literal_eval)
