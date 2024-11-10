import collections
import math

import numpy as np
import pandas as pd

import config
import data
import util


def get_text_des(text, raw_text, model, pooling_type=0):
    n = len(text)

    origin_text_rep = util.get_text_rep(text, model, pooling_type)
    result = [0.0] * n
    for idx in range(n):
        if result[idx] != 0:
            continue
        token = text[idx]
        token_idx = util.get_index(token, text)
        # delete the token from the original text
        tmp_text = str(raw_text).strip().split()
        tmp_text = np.delete(tmp_text, token_idx).tolist()
        drop_token_text_rep = util.get_text_rep(tmp_text, model, pooling_type)

        des_value = 1 - util.compute_similarity(origin_text_rep, drop_token_text_rep)
        for t_idx in token_idx:
            result[t_idx] = des_value

    surround_result = []
    text_str = ' '.join(text)
    for idx in range(n):
        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = ' '.join(text[left: right])
        tmp_str = [token for token in text_str.replace(tmp_str, '').split() if token.strip() != '']
        if len(tmp_str) == 0:
            surround_result.append(1)
            continue

        if len(tmp_str) == 1:
            drop_token_text_rep = util.get_token_rep(tmp_str[0], model)
        else:
            drop_token_text_rep = util.get_text_rep(tmp_str, model, pooling_type)
        des_value = 1 - util.compute_similarity(origin_text_rep, drop_token_text_rep)

        surround_result.append(des_value)

    return result, surround_result


def get_title_sim(bug_text, bug_id, model, bug_title, pooling_type=1):

    n = len(bug_text)
    title_rep = util.get_text_rep(bug_title, model, pooling_type)

    result = []
    surround_result = []
    for idx in range(n):
        token = bug_text[idx]
        token_rep = util.get_token_rep(token, model)

        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = bug_text[left: right]
        phrase_rep = util.get_text_rep(tmp_str, model, pooling_type)

        sim_val = util.compute_similarity(title_rep, token_rep)
        result.append(sim_val)

        sim_val = util.compute_similarity(title_rep, phrase_rep)
        surround_result.append(sim_val)

    return result, surround_result



def get_history_sim(bug_id, used_bug_id, text, history_data, keyword_data, bug_target_code_data):

    m = len(text)

    tf_idf_result = [0.0] * m
    idf_result = [0.0] * m
    max_freq_result = [0.0] * m
    mean_freq_result = [0.0] * m
    median_freq_result = [0.0] * m
    key_time_result = [0.0] * m
    if str(bug_id) in history_data.keys():
        history_content = history_data[str(bug_id)]
    else:
        print("bug id not in history data")
        return tf_idf_result, idf_result, max_freq_result, mean_freq_result, median_freq_result, key_time_result

    # get the content of buggy code files corresponding to historical bug reports
    history_count = 0
    history_idx = 0
    target_code = []
    history_keyword = []
    while history_idx < len(history_content) and history_count < config.top_n:
        history_item = history_content[history_idx]
        history_id = int(history_item[0])

        if history_id not in used_bug_id:
            history_idx += 1
            continue

        keyword_df = keyword_data[keyword_data['bugId'] == history_id]
        keywords = str(keyword_df['keywords'].values[0]).strip().split()
        history_keyword.append(dict(collections.Counter(keywords)))

        bug_target_code = bug_target_code_data[bug_target_code_data['bugId'] == history_id]
        buggy_files = str(bug_target_code['targetCodeToken'].values[0]).strip().split(';')
        for file in buggy_files:
            if file != '':
                file = file.strip().split()
                target_code.append(file)
        history_count += 1
        history_idx += 1

        bug_counter = dict(collections.Counter(text))
        code_counters = []
        for code in target_code:
            code_counter = dict(collections.Counter(code))
            code_counters.append(code_counter)
        n = len(target_code)
        for idx in range(m):
            token = text[idx]

            idf = 0
            freq = []
            key_time = 0

            for code in code_counters:
                if token in code.keys():
                    idf += 1
                    tf = code[token] / len(code)
                else:
                    tf = 0
                freq.append(tf)

            for keywords in history_keyword:
                if token in keywords.keys():
                    key_time += 1

            df = idf
            idf_result[idx] = df
            idf = math.log(n / (idf + 1))
            tf_idf_result[idx] = (bug_counter[token] / m) * idf

            freq = np.array(freq)

            max_freq_result[idx] = max(np.max(freq, axis=0), max_freq_result[idx])
            mean_freq_result[idx] = max(np.mean(freq, axis=0), mean_freq_result[idx])
            median_freq_result[idx] = max(np.median(freq, axis=0), median_freq_result[idx])
            key_time_result[idx] += key_time

    return tf_idf_result, idf_result, max_freq_result, mean_freq_result, median_freq_result, key_time_result


def get_feedback_data(bug_id, bug_text):
    feedback_code_list = data.read_pseudo_feedback(bug_id)

    code_corpus_filepath = config.code_filepath_prefix + str(bug_id) + "\\CodeCorpus.txt"

    code_data = []

    with open(code_corpus_filepath, 'r', errors='ignore') as f:
        count = 0

        line = f.readline()
        while line != '':
            line = line.strip().split('\t')
            code_id = int(line[0])
            if code_id in feedback_code_list:
                count += 1
                code_text = str(line[2]).strip().split()
                code_data.append(code_text)

                if count == len(feedback_code_list):
                    break
            line = f.readline()

    idf_result = []
    tf_idf_result = []
    max_freq_result = []
    mean_freq_result = []
    median_freq_result = []

    bug_counter = dict(collections.Counter(bug_text))
    code_counters = []
    for code in code_data:
        code_counter = dict(collections.Counter(code))
        code_counters.append(code_counter)

    n = len(bug_text)
    m = len(code_data)
    for token in bug_text:
        idf = 0
        freq = []
        for code in code_counters:
            if token in code.keys():
                idf += 1
                tf = code[token] / len(code)
            else:
                tf = 0
            freq.append(tf)

        df = idf
        idf_result.append(df)
        idf = math.log(m / (idf + 1))
        tf_idf_result.append((bug_counter[token] / n) * idf)

        freq = np.array(freq)

        max_freq_result.append(np.max(freq, axis=0))
        mean_freq_result.append(np.mean(freq, axis=0))
        median_freq_result.append(np.median(freq, axis=0))

    return idf_result, tf_idf_result, max_freq_result, mean_freq_result, median_freq_result


def get_feedback_freq(text, bug_id):
    idf_result, tf_idf_result, max_freq_result, mean_freq_result, median_freq_result = get_feedback_data(bug_id, text)

    return idf_result, tf_idf_result, max_freq_result, mean_freq_result, median_freq_result
