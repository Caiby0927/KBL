import math

import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from gensim import utils
from numpy import float32 as REAL

import config

ps = PorterStemmer()
ENG_STOP_WORDS_SET = set()
for word in stopwords.words('english'):
    word = word.lower().strip()
    word = ps.stem(word)
    ENG_STOP_WORDS_SET.add(word)


def get_index(item, lst):
    return [index for (index, value) in enumerate(lst) if value == item]

def get_text_rep(text, model, pooling_type=1):
    emb_matrix = []

    if len(text) == 1:
        return get_token_rep(text[0], model)

    for token in text:
        vector = get_token_rep(token, model)
        emb_matrix.append(vector)

    emb_matrix = np.array(emb_matrix)

    pooling_res = []
    if pooling_type == 0:
        for col in np.nditer(emb_matrix, order="F", flags=['external_loop']):
            col = list(col)
            max_value = max(col)
            pooling_res.append(max_value)
    else:
        for col in np.nditer(emb_matrix, order='F', flags=['external_loop']):
            col = list(col)
            mean_value = np.mean(col)
            pooling_res.append(mean_value)

    text_rep = np.array(pooling_res)
    return text_rep


def get_token_rep(token, model):
    try:
        vector = model.get_vector(token)
    except:
        vector = [0.0] * config.emb_size

    return np.array(vector)


def compute_similarity(vec1, vec2):
    a = vec1.dot(vec2)
    b = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if b == 0:
        sim_value = 0
    else:
        sim_value = a / b
    return sim_value


def compute_idf(n, df):
    return math.log(n / (df + 1))


def compute_tf_idf(n, count, idf):
    return (count / n) * idf


def process_sent(sent):
    sent = sent.strip().split()
    tokens = []

    for index in range(len(sent)):
        token = sent[index]
        word_buf = []
        for character in token:
            if ('a' <= character <= 'z') or ('A' <= character <= 'Z'):
                word_buf.append(character)
                continue

            split_tokens = split_camel_case(word_buf)
            tokens += split_tokens
            word_buf.clear()

        if len(word_buf) != 0:
            split_tokens = split_camel_case(word_buf)
            tokens += split_tokens

    return preprocess(tokens)


def preprocess(tokens):

    ps = PorterStemmer()
    ENG_STOP_WORDS_SET = set()
    for word in stopwords.words('english'):
        word = word.lower().strip()
        word = ps.stem(word)
        ENG_STOP_WORDS_SET.add(word)

    res = []

    for index in range(len(tokens)):
        token = tokens[index].lower().strip()
        token = ps.stem(token)
        if token not in ENG_STOP_WORDS_SET:
            res.append(token)

    return res


def split_camel_case(word_buf):

    tokens = []

    word_buf = ''.join(filter(str.isalpha, word_buf))

    length = len(word_buf)

    if length == 1:
        return []

    if length != 0:
        k = 0
        i = 0
        j = 1

        while i < length - 1 and j < length:
            first = word_buf[i]
            second = word_buf[j]
            if ('A' <= first <= 'Z') and ('a' <= second <= 'z'):
                token = ''.join(word_buf[k: i]).strip()
                if token != "" and len(token) > 0:
                    tokens.append(token)
                k = i
                i += 1
                j += 1
                continue
            if ('a' <= first <= 'z') and ('A' <= second <= 'Z'):
                token = ''.join(word_buf[k: j]).strip()
                if token != "" and len(token) > 0:
                    tokens.append(token)
                k = j
            i += 1
            j += 1

        if k < length:
            token = ''.join(word_buf[k:])
            if token.strip() != "" and len(token) > 0:
                tokens.append(token)

    return tokens


def left_right_boundary(c_pos, max_len, window):
    bottom = c_pos - window
    top = c_pos + window + 1
    if bottom < 0:
        bottom = 0
    if top >= max_len:
        top = max_len
    return bottom, top


def model2dict(model):
    model_dict = {}
    for index in tqdm(model.wv.index_to_key):
        key = index[0]
        value = model.wv.get_vector(index)
        model_dict[key] = value

    return model_dict


def save_model_format(save_filename, model_keyed_vector, binary=True):
    total_vec = len(model_keyed_vector.index_to_key)
    with utils.open(save_filename, 'wb') as f:
        f.write(utils.to_utf8("%s %s\n" % (total_vec, model_keyed_vector.vector_size)))
        for key in model_keyed_vector.index_to_key:
            value = model_keyed_vector.get_vector(key)
            if binary:
                value = value.astype(REAL)
                f.write(utils.to_utf8(key) + b" " + value.tobytes())
            else:
                f.write(utils.to_utf8("%s %s\n" % (key, ' '.join(repr(val) for val in value))))

