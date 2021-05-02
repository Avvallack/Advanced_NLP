import json
import pickle
import string
import pandas as pd
from difflib import SequenceMatcher


def prepare_labeled_dataframe(data_path: str = './data/recognized_dict.pickle'):
    with open(data_path, 'rb') as f:
        recognized_texts = pickle.load(f)

    s_dict = dict()
    for k in recognized_texts.keys():
        try:
            with open('.' + k + '.txt', 'r') as f:
                s_dict[k] = json.load(f)
        except FileNotFoundError:
            continue

    df = pd.DataFrame(columns=['name', 'recognized_text'])
    df['name'] = recognized_texts.keys()
    df['recognized_text'] = recognized_texts.values()
    df2 = pd.DataFrame(columns=['name', 'target'])
    df2['name'] = s_dict.keys()
    df2['target'] = s_dict.values()
    merged = df.merge(df2, on='name')
    merged['recognized_text'] = merged['recognized_text'].apply(
        lambda x: (' '.join(x)).lower().translate(str.maketrans('', '', string.punctuation)))
    merged['cleaned_target'] = merged['target'].apply(lambda x: target_prp(x))
    rest = []

    for i, qr in merged.iterrows():
        rest.append(label_texts(qr['recognized_text'], qr['cleaned_target']))

    merged['label_boxes'] = rest
    merged['tokenized_texts'] = merged['recognized_text'].apply(lambda x: x.split())

    finals = []
    for text, dct in merged[['tokenized_texts', 'label_boxes']].values:
        finals.append(prep_df(text, dct))
    merged['labels'] = finals

    return merged[['tokenized_texts', 'labels']]


def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + query_len - step <= len(corpus):
            match_values.append(_match(query, corpus[m: m-1+query_len]))
            if verbose:
                print(query, "-", corpus[m: m + query_len], _match(query, corpus[m: m + query_len]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + query_len] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l: bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    query_len = len(query)

    if flex >= query_len/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip().split(), match_value


def target_prp(target_dict):
    for key in target_dict.keys():
        target_dict[key] = target_dict[key].lower().translate(str.maketrans('', '', string.punctuation))
    return target_dict


def label_texts(text, target_dict):
    res_dict = dict()
    for key in target_dict.keys():
        res_dict[key] = get_best_match(target_dict[key], text)
    return res_dict


def prep_df(tokenized_text, target_dict):
    labeling = ['O'] * len(tokenized_text)
    for i, token in enumerate(tokenized_text):
        for key in target_dict.keys():
            if token in target_dict[key][0]:
                labeling[i] = key
    return labeling


if __name__ == '__main__':
    data_set = prepare_labeled_dataframe()
    with open('./data/prepared/dataset.pickle', 'wb') as f_out:
        pickle.dump(data_set, f_out)
