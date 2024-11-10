import collections
import math
import random

import jpype
import numpy as np

import pandas as pd

import config


def convert_score_2_label(model_name):
    test_tokens = pd.read_csv(config.all_test_tokens_filepath)
    scores = []
    labels = []

    with open(config.res_filepath_prefix + "\\predict_result_" + model_name + ".txt", 'r') as f:
        line = f.readline()
        while line != '':
            score = float(line.strip())
            if score > 0.5:
                labels.append(1)
            else:
                labels.append(0)
            scores.append(score)
            line = f.readline()

    if len(scores) < len(test_tokens):
        scores.extend([0] * (len(test_tokens) - len(scores)))
        labels.extend([0] * (len(test_tokens) - len(labels)))

    test_tokens.insert(test_tokens.shape[1], 'predict_score', scores)
    test_tokens.insert(test_tokens.shape[1], 'predict_label', labels)

    test_tokens.to_csv(config.res_filepath_prefix + "\\test_result_" + model_name + ".csv", index=False)


def merge_tokens(model, history_path=None, low_quality_path=None, add_after=0, drop_time=None, res_file=None, project=None):

    if history_path is not None:
        global history_data
        history_data = get_history_keyword(history_path)

    if low_quality_path is not None:
        global low_quality_tokens
        low_quality_tokens = read_low_quality_tokens(low_quality_path)

    if project is None:
        test_result = pd.read_csv(config.res_filepath_prefix + "\\test_result_" + model + ".csv")
    else:
        test_result = pd.read_csv("E:\\query-reformulation\\processed_dataset\\" + project + "\\test_result_" + model + ".csv")
    bug_id_set = set(test_result['bugId'].tolist())

    predict_result = []
    for bug_id in bug_id_set:
        df = test_result[test_result['bugId'] == bug_id]
        keyword = []
        predict_keyword = []
        for index, row in df.iterrows():
            token = row['token']
            if type(token) is not str:
                if math.isnan(token):
                    token = 'null'
            if row['label'] == 1:
                keyword.append(token.strip())
            if row['predict_label'] == 1 and token not in low_quality_tokens:
                predict_keyword.append(token)
        keyword = ' '.join(keyword).strip()

        global repeat_times
        predict_keyword = drop_dup(predict_keyword, repeat_times)
        if add_after == 0:
            predict_keyword += add_history_keyword(bug_id)
            predict_keyword = drop_dup(predict_keyword, drop_time)
        elif add_after == 1:
            predict_keyword = drop_dup(predict_keyword, drop_time)
            predict_keyword += add_history_keyword(bug_id)
        elif add_after == 2:
            predict_keyword = drop_dup(predict_keyword, drop_time)
        elif add_after == 3:
            predict_keyword += add_history_keyword(bug_id)

        predict_keyword = ' '.join(predict_keyword).strip()

        if len(predict_keyword) == 0:
            if model == 'lightgbm':
                predict_keyword = deal_with_no_keyword(bug_id, model)
            else:
                predict_keyword = deal_with_no_keyword_randomly(bug_id, model)

        predict_result.append([bug_id, keyword, predict_keyword])

    res = pd.DataFrame(predict_result, columns=['bugId', 'keywords', 'predict_keywords'])
    if res_file is None:
        res.to_csv(config.res_filepath_prefix + "\\predict_result_" + model + ".csv", index=False)
    else:
        res.to_csv(res_file, index=False)


def drop_dup(token_lst, drop_time=None):
    token_counter = dict(collections.Counter(token_lst))
    keyword_lst = []
    for key, val in token_counter.items():
        if drop_time is None:
            if val > config.keyword_dup_constraint:
                keyword_lst += [key] * config.keyword_dup_constraint
            else:
                keyword_lst += [key] * val
        else:
            # print(drop_time)
            if val > drop_time:
                keyword_lst += [key] * drop_time
            else:
                keyword_lst += [key] * val

    return keyword_lst


def deal_with_no_keyword(bug_id, model):
    test_result = pd.read_csv(config.filepath_prefix + "\\test_result_" + model + ".csv",
                              usecols=['bugId', 'token', 'predict_score'])
    test_result = test_result[test_result['bugId'] == bug_id]
    test_result = test_result.sort_values(by=['predict_score'], ascending=False)
    if len(test_result) > 10:
        test_result = test_result.head(10)

    keyword = []
    for index, row in test_result.iterrows():
        token = row['token']
        if type(token) is not str:
            token = 'null'
        keyword.append(token)
    keyword = ' '.join(keyword).strip()
    return keyword


def deal_with_no_keyword_randomly(bug_id, model):
    test_result = pd.read_csv(config.res_filepath_prefix + "\\test_result_" + model + ".csv",
                              usecols=['bugId', 'token'])
    test_result = test_result[test_result['bugId'] == bug_id]

    if len(test_result) > 10:
        test_result = test_result.sample(n=10)

    keyword = []
    for index, row in test_result.iterrows():
        token = row['token']
        if type(token) is not str:
            token = 'null'
        keyword.append(token)
    keyword = ' '.join(keyword).strip()
    return keyword


def read_low_quality_tokens(low_quality_file=None):
    low_quality = []
    if low_quality_file is None:
        with open(config.filepath_prefix + "\\data\\project_level_low_quality_tokens.txt", 'r') as f:
            line = f.readline()
            while line != '':
                low_quality.append(line.strip())
                line = f.readline()
    else:
        with open(low_quality_file) as f:
            line = f.readline()
            while line != '':
                low_quality.append(line.strip())
                line = f.readline()

    return low_quality


def get_history_keyword(path=None):
    if path is None:
        history_dataset = pd.read_csv(config.share_keyword_file)
    else:
        print("get history data from " + path)
        history_dataset = pd.read_csv(path)
    return history_dataset


def add_history_keyword(bug_id):
    history_df = history_data[history_data['bugId'] == bug_id]
    share_keyword = history_df['sharedKeywords']
    if len(share_keyword) != 0:
        share_keyword = str(share_keyword.values[0]).strip().split()
    else:
        share_keyword = []

    res = []
    for keyword in share_keyword:
        if keyword not in low_quality_tokens:
            res.append(keyword)
    share_keyword = res

    return share_keyword


def evaluate(model_name, repeat_time=None):

    if str(model_name).startswith("lightgbm"):
        convert_score_2_label(model_name)

    merge_tokens(model_name)


def compute_result(model_name, res_file=None):
    if res_file is None:
        result_data = pd.read_csv(config.res_filepath_prefix + "\\predict_rank_result_" + model_name + ".csv")
    else:
        result_data = pd.read_csv(res_file)

    n = len(result_data)
    print(n)

    eff_result = result_data['effectiveness'].tolist()
    # eff_result = result_data['Effectiveness'].tolist()
    eff_counter = dict(collections.Counter(eff_result))
    eff_1 = 0
    eff_5 = 0
    eff_10 = 0
    eff_20 = 0
    for eff in eff_counter.keys():
        if eff == 1:
            eff_1 += eff_counter.get(eff)
        if 1 <= eff <= 5:
            eff_5 += eff_counter.get(eff)
        if 1 <= eff <= 10:
            eff_10 += eff_counter.get(eff)
        if 1 <= eff <= 20:
            eff_20 += eff_counter.get(eff)

    print(eff_1, eff_5, eff_10, eff_20)
    print(eff_1 / n, eff_5 / n, eff_10 / n, eff_20 / n)
    eff_res = str(eff_1 / n) + ", " + str(eff_5 / n) + ", " + str(eff_10 / n) + ", " + str(eff_20 / n)

    map_result = np.sum(result_data['AP'].tolist()) / n
    # map_result = np.mean(result_data['AP'].tolist())
    print(map_result)

    mrr_result = np.sum(result_data['RR'].tolist()) / n
    # mrr_result = np.mean(result_data['RR'].tolist())
    print(mrr_result)

    return eff_res, str(map_result), str(mrr_result)


repeat_times = None

history_data = get_history_keyword()
low_quality_tokens = read_low_quality_tokens()
