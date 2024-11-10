import collections

import pandas as pd
import numpy as np

import config
import data
import util


def get_code_corpus_rep(bug_id, model, pooling_type):
    code_corpus_filepath = config.code_filepath_prefix + str(bug_id) + "\\CodeCorpus.txt"
    code_rep_data = {}

    with open(code_corpus_filepath, 'r', errors='ignore') as f:
        line = f.readline()
        while line != '':
            line = line.split('\t')
            code_id = int(line[0])
            code_text = line[2].strip()
            if code_text != '':
                code_text = code_text.split()
                if len(code_text) > 1:
                    code_rep = util.get_text_rep(code_text, model, pooling_type)
                else:
                    code_rep = util.get_token_rep(code_text[0], model)
                code_rep_data[code_id] = code_rep

            line = f.readline()

    return code_rep_data


def text_sim(text, bug_id, model, pooling_type=0):

    bug_rep = util.get_text_rep(text, model, pooling_type)

    code_corpus_filepath = config.code_filepath_prefix + str(bug_id) + "\\CodeCorpus.txt"
    bug_code_sim = {}
    code_data = []

    with open(code_corpus_filepath, 'r', errors='ignore') as f:
        line = f.readline()
        while line != '':
            line = line.split('\t')
            code_id = int(line[0])
            code_text = line[2].strip()
            if code_text != '':
                code_text = code_text.split()
                if len(code_text) > 1:
                    code_rep = util.get_text_rep(code_text, model, pooling_type)
                else:
                    code_rep = util.get_token_rep(code_text[0], model)

                sim_value = util.compute_similarity(bug_rep, code_rep)
            else:
                sim_value = 0.0
            bug_code_sim[code_id] = sim_value
            code_data.append(code_text)

            line = f.readline()

    bug_code_sim = sorted(bug_code_sim.items(), key=lambda item: (item[1], item[0]), reverse=True)[: config.top_n]
    code_files = []
    for tup in bug_code_sim:
        code_id = tup[0]
        code_text = code_data[code_id]
        code_files.append(dict(collections.Counter(code_text)))
    code_data.clear()

    tf_idf_result = []
    idf_result = []
    max_freq_result = []
    mean_freq_result = []
    median_freq_result = []

    n = len(text)
    m = len(code_files)
    bug_counter = dict(collections.Counter(text))
    for token in text:
        idf = 0
        freq = []

        for code in code_files:

            if token in code.keys():
                idf += 1
                tf = code[token] / m
            else:
                tf = 0.0
            freq.append(tf)

        df = idf
        idf_result.append(df)
        idf = util.compute_idf(m, idf)
        tf_idf_result.append(util.compute_tf_idf(n, bug_counter[token], idf))

        freq = np.array(freq)

        max_freq_result.append(np.max(freq, axis=0))
        mean_freq_result.append(np.mean(freq, axis=0))
        median_freq_result.append(np.median(freq, axis=0))

    return idf_result, tf_idf_result, max_freq_result, mean_freq_result, median_freq_result


def code_sim(text, bug_id, model, pooling_type=0):
    n = len(text)

    code_rep_data = get_code_corpus_rep(bug_id, model, pooling_type)

    max_sim_result = []
    mean_sim_result = []
    median_sim_result = []

    surround_max_sim_result = []
    surround_mean_sim_result = []
    surround_median_sim_result = []
    for idx in range(n):
        token = text[idx]
        token_rep = util.get_token_rep(token, model)

        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = text[left: right]
        phrase_rep = util.get_text_rep(tmp_str, model, pooling_type)

        token_code_similarity = []
        surround_code_similarity = []
        for code_id, code_rep in code_rep_data.items():
            sim_value = util.compute_similarity(token_rep, code_rep)

            token_code_similarity.append(sim_value)
            sim_value = util.compute_similarity(phrase_rep, code_rep)
            surround_code_similarity.append(sim_value)

        token_code_similarity = np.array(token_code_similarity)
        surround_code_similarity = np.array(surround_code_similarity)

        max_sim_result.append(np.max(token_code_similarity, axis=0))
        mean_sim_result.append(np.mean(token_code_similarity, axis=0))
        median_sim_result.append(np.median(token_code_similarity, axis=0))

        surround_max_sim_result.append(np.max(surround_code_similarity, axis=0))
        surround_mean_sim_result.append(np.mean(surround_code_similarity, axis=0))
        surround_median_sim_result.append(np.median(surround_code_similarity, axis=0))

    return max_sim_result, mean_sim_result, median_sim_result, surround_max_sim_result, surround_mean_sim_result, surround_median_sim_result


def class_sim(text, bug_id, model, pooling_type=0):

    n = len(text)

    classname_list = data.get_class_name(bug_id)
    classname_rep_list = []
    for classname in classname_list:
        classname = util.split_camel_case(classname)
        if len(classname) == 0:
            continue

        util.preprocess(classname)

        if len(classname) > 1:
            classname_rep = util.get_text_rep(classname, model, pooling_type)
        else:
            classname_rep = util.get_token_rep(classname[0], model)

        classname_rep_list.append(classname_rep)

    max_sim_result = []
    mean_sim_result = []
    median_sim_result = []

    surround_max_sim_result = []
    surround_mean_sim_result = []
    surround_median_sim_result = []
    for idx in range(n):
        token = text[idx]
        token_rep = util.get_token_rep(token, model)

        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = text[left: right]
        phrase_rep = util.get_text_rep(tmp_str, model, pooling_type)

        token_classname_sim_list = []
        surround_classname_similarity = []
        for classname_rep in classname_rep_list:
            sim_value = util.compute_similarity(token_rep, classname_rep)
            token_classname_sim_list.append(sim_value)
            sim_value = util.compute_similarity(phrase_rep, classname_rep)
            surround_classname_similarity.append(sim_value)

        token_classname_sim_list = np.array(token_classname_sim_list)
        surround_classname_similarity = np.array(surround_classname_similarity)

        max_sim_result.append(np.max(token_classname_sim_list, axis=0))
        mean_sim_result.append(np.mean(token_classname_sim_list, axis=0))
        median_sim_result.append(np.median(token_classname_sim_list, axis=0))

        surround_max_sim_result.append(np.max(surround_classname_similarity, axis=0))
        surround_mean_sim_result.append(np.mean(surround_classname_similarity, axis=0))
        surround_median_sim_result.append(np.median(surround_classname_similarity, axis=0))

    return max_sim_result, mean_sim_result, median_sim_result, surround_max_sim_result, surround_mean_sim_result, surround_median_sim_result


def feedback_sim(text, bug_id, model, pooling_type=0):
    n = len(text)

    feedback_code_list = data.read_pseudo_feedback(bug_id)

    code_corpus_filepath = config.code_filepath_prefix + str(bug_id) + "\\CodeCorpus.txt"
    code_rep_data = []

    with open(code_corpus_filepath, 'r', errors='ignore') as f:
        count = 0
        line = f.readline()
        while line != '':
            line = line.split('\t')
            code_id = int(line[0])
            if code_id in feedback_code_list:
                count += 1
                code_text = line[2].strip()
                if code_text != '':
                    code_text = code_text.split()
                    code_rep = util.get_text_rep(code_text, model, pooling_type)
                    code_rep_data.append(code_rep)
                if count == len(feedback_code_list):
                    break

            line = f.readline()

    max_sim_result = []
    mean_sim_result = []
    median_sim_result = []

    surround_max_sim_result = []
    surround_mean_sim_result = []
    surround_median_sim_result = []
    for idx in range(n):
        token = text[idx]
        token_rep = util.get_token_rep(token, model)

        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = text[left: right]
        phrase_rep = util.get_text_rep(tmp_str, model, pooling_type)

        token_code_similarity = []
        surround_code_similarity = []
        for code_rep in code_rep_data:
            sim_value = util.compute_similarity(token_rep, code_rep)
            token_code_similarity.append(sim_value)
            sim_value = util.compute_similarity(phrase_rep, code_rep)
            surround_code_similarity.append(sim_value)

        token_code_similarity = np.array(token_code_similarity)
        surround_code_similarity = np.array(surround_code_similarity)

        max_sim_result.append(np.max(token_code_similarity, axis=0))
        mean_sim_result.append(np.mean(token_code_similarity, axis=0))
        median_sim_result.append(np.median(token_code_similarity, axis=0))

        surround_max_sim_result.append(np.max(surround_code_similarity, axis=0))
        surround_mean_sim_result.append(np.mean(surround_code_similarity, axis=0))
        surround_median_sim_result.append(np.median(surround_code_similarity, axis=0))

    return max_sim_result, mean_sim_result, median_sim_result, surround_max_sim_result, surround_mean_sim_result, surround_median_sim_result


def feedback_classname_sim(text, bug_id, model, for_train=True, pooling_type=0):
    n = len(text)

    feedback_code_list = data.read_pseudo_feedback(bug_id)

    code_corpus_filepath = config.code_filepath_prefix + str(bug_id) + "\\CodeCorpus.txt"
    classname_rep_list = []

    with open(code_corpus_filepath, 'r', errors='ignore') as f:
        count = 0
        line = f.readline()
        while line != '':
            line = line.split('\t')
            code_id = int(line[0])
            if code_id in feedback_code_list:
                count += 1
                print(line[1])
                classname = line[1].split('\\')[-1][: -5]
                classname = util.split_camel_case(classname)
                if len(classname) == 0:
                    line = f.readline()
                    continue

                classname = util.preprocess(classname)
                print(classname)

                if len(classname) == 0:
                    print(line[1])
                    line = f.readline()
                    continue

                if len(classname) > 1:
                    classname_rep = util.get_text_rep(classname, model, pooling_type)
                else:
                    classname_rep = util.get_token_rep(classname[0], model)

                classname_rep_list.append(classname_rep)
                if count == len(feedback_code_list):
                    break

            line = f.readline()

    max_sim_result = []
    mean_sim_result = []
    median_sim_result = []

    surround_max_sim_result = []
    surround_mean_sim_result = []
    surround_median_sim_result = []
    for idx in range(n):
        token = text[idx]
        token_rep = util.get_token_rep(token, model)

        left, right = util.left_right_boundary(idx, n, config.win_size)
        tmp_str = text[left: right]
        phrase_rep = util.get_text_rep(tmp_str, model, pooling_type)

        token_feedback_classname_similarity = []
        surround_classname_similarity = []
        for classname_rep in classname_rep_list:
            sim_value = util.compute_similarity(token_rep, classname_rep)
            token_feedback_classname_similarity.append(sim_value)
            sim_value = util.compute_similarity(phrase_rep, classname_rep)
            surround_classname_similarity.append(sim_value)

        token_feedback_classname_similarity = np.array(token_feedback_classname_similarity)
        surround_classname_similarity = np.array(surround_classname_similarity)

        max_sim_result.append(np.max(token_feedback_classname_similarity, axis=0))
        mean_sim_result.append(np.mean(token_feedback_classname_similarity, axis=0))
        median_sim_result.append(np.median(token_feedback_classname_similarity, axis=0))

        surround_max_sim_result.append(np.max(surround_classname_similarity, axis=0))
        surround_mean_sim_result.append(np.mean(surround_classname_similarity, axis=0))
        surround_median_sim_result.append(np.median(surround_classname_similarity, axis=0))

    return max_sim_result, mean_sim_result, median_sim_result, surround_max_sim_result, surround_mean_sim_result, surround_median_sim_result
