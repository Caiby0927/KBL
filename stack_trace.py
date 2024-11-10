import time

import bug_semantic
import code_semantic
import text_features

import re

import pandas as pd

import config

from xml.dom.minidom import parse
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet, stopwords

import data
import util

from gensim.models import Word2Vec, FastText


# EngStopWord = ["a", "a's", "able", "about", "above",
#                "according", "accordingly", "across", "actually", "after",
#                "afterwards", "again", "against", "ain't", "all", "allow",
#                "allows", "almost", "alone", "along", "already", "also",
#                "although", "always", "am", "among", "amongst", "an", "and",
#                "another", "any", "anybody", "anyhow", "anyone", "anything",
#                "anyway", "anyways", "anywhere", "apart", "appear",
#                "appreciate", "appropriate", "are", "aren't", "around", "as",
#                "aside", "ask", "asking", "associated", "at", "available",
#                "away", "awfully", "b", "be", "became", "because", "become",
#                "becomes", "becoming", "been", "before", "beforehand",
#                "behind", "being", "believe", "below", "beside", "besides",
#                "best", "better", "between", "beyond", "both", "brief", "but",
#                "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot",
#                "cant", "cause", "causes", "certain", "certainly", "changes",
#                "clearly", "co", "com", "come", "comes", "concerning",
#                "consequently", "consider", "considering", "contain",
#                "containing", "contains", "corresponding", "could", "couldn't",
#                "course", "currently", "d", "definitely", "described",
#                "despite", "did", "didn't", "different", "do", "does",
#                "doesn't", "doing", "don't", "done", "down", "downwards",
#                "during", "e", "each", "edu", "eg", "eight", "either", "else",
#                "elsewhere", "enough", "entirely", "especially", "et", "etc",
#                "even", "ever", "every", "everybody", "everyone", "everything",
#                "everywhere", "ex", "exactly", "example", "except", "f", "far",
#                "few", "fifth", "first", "five", "followed", "following",
#                "follows", "for", "former", "formerly", "forth", "four",
#                "from", "further", "furthermore", "g", "get", "gets",
#                "getting", "given", "gives", "go", "goes", "going", "gone",
#                "got", "gotten", "greetings", "h", "had", "hadn't", "happens",
#                "hardly", "has", "hasn't", "have", "haven't", "having", "he",
#                "he's", "hello", "help", "hence", "her", "here", "here's",
#                "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
#                "hi", "him", "himself", "his", "hither", "hopefully", "how",
#                "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
#                "if", "ignored", "immediate", "in", "inasmuch", "inc",
#                "indeed", "indicate", "indicated", "indicates", "inner",
#                "insofar", "instead", "into", "inward", "is", "isn't", "it",
#                "it'd", "it'll", "it's", "its", "itself", "j", "just", "k",
#                "keep", "keeps", "kept", "know", "knows", "known", "l", "last",
#                "lately", "later", "latter", "latterly", "least", "less",
#                "lest", "let", "let's", "like", "liked", "likely", "little",
#                "look", "looking", "looks", "ltd", "m", "mainly", "many",
#                "may", "maybe", "me", "mean", "meanwhile", "merely", "might",
#                "more", "moreover", "most", "mostly", "much", "must", "my",
#                "myself", "n", "name", "namely", "nd", "near", "nearly",
#                "necessary", "need", "needs", "neither", "never",
#                "nevertheless", "new", "next", "nine", "no", "nobody", "non",
#                "none", "noone", "nor", "normally", "not", "nothing", "novel",
#                "now", "nowhere", "o", "obviously", "of", "off", "often", "oh",
#                "ok", "okay", "old", "on", "once", "one", "ones", "only",
#                "onto", "or", "other", "others", "otherwise", "ought", "our",
#                "ours", "ourselves", "out", "outside", "over", "overall",
#                "own", "p", "particular", "particularly", "per", "perhaps",
#                "placed", "please", "plus", "possible", "presumably",
#                "probably", "provides", "q", "que", "quite", "qv", "r",
#                "rather", "rd", "re", "really", "reasonably", "regarding",
#                "regardless", "regards", "relatively", "respectively", "right",
#                "s", "said", "same", "saw", "say", "saying", "says", "second",
#                "secondly", "see", "seeing", "seem", "seemed", "seeming",
#                "seems", "seen", "self", "selves", "sensible", "sent",
#                "serious", "seriously", "seven", "several", "shall", "she",
#                "should", "shouldn't", "since", "six", "so", "some",
#                "somebody", "somehow", "someone", "something", "sometime",
#                "sometimes", "somewhat", "somewhere", "soon", "sorry",
#                "specified", "specify", "specifying", "still", "sub", "such",
#                "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
#                "th", "than", "thank", "thanks", "thanx", "that", "that's",
#                "thats", "the", "their", "theirs", "them", "themselves",
#                "then", "thence", "there", "there's", "thereafter", "thereby",
#                "therefore", "therein", "theres", "thereupon", "these", "they",
#                "they'd", "they'll", "they're", "they've", "think", "third",
#                "this", "thorough", "thoroughly", "those", "though", "three",
#                "through", "throughout", "thru", "thus", "to", "together",
#                "too", "took", "toward", "towards", "tried", "tries", "truly",
#                "try", "trying", "twice", "two", "u", "un", "under",
#                "unfortunately", "unless", "unlikely", "until", "unto", "up",
#                "upon", "us", "use", "used", "useful", "uses", "using",
#                "usually", "uucp", "v", "value", "various", "very", "via",
#                "viz", "vs", "w", "want", "wants", "was", "wasn't", "way",
#                "we", "we'd", "we'll", "we're", "we've", "welcome", "well",
#                "went", "were", "weren't", "what", "what's", "whatever",
#                "when", "whence", "whenever", "where", "where's", "whereafter",
#                "whereas", "whereby", "wherein", "whereupon", "wherever",
#                "whether", "which", "while", "whither", "who", "who's",
#                "whoever", "whole", "whom", "whose", "why", "will", "willing",
#                "wish", "with", "within", "without", "won't", "wonder",
#                "would", "would", "wouldn't", "x", "y", "yes", "yet", "you",
#                "you'd", "you'll", "you're", "you've", "your", "yours",
#                "yourself", "yourselves", "z", "zero", "quot"]
#
ps = PorterStemmer()
ENG_STOP_WORDS_SET = set()
for word in stopwords.words('english'):
    word = word.lower().strip()
    word = ps.stem(word)
    ENG_STOP_WORDS_SET.add(word)


def is_stack_trace():
    dom_tree = parse(config.origin_dataset_xml_filepath)
    root_node = dom_tree.documentElement

    bugs = root_node.getElementsByTagName("table")

    pattern = re.compile(config.regular_exp)

    result_list = []

    used_bug_id = data.get_used_br()

    history_data = data.get_history_data()

    keyword_data = data.get_keyword_data()

    bug_target_code_data = pd.read_csv(config.processed_filepath_prefix + "//BugTargetCode.csv")

    code_model = Word2Vec().wv.load_word2vec_format(config.embedding_model_path + "word2vec\\train_bug_model_format.txt", binary=True)

    count = 0

    for bug in bugs:
        if count >= 64:
            break

        fields = bug.getElementsByTagName("column")

        bug_id = ""
        summary = ""
        description = ""

        for field in fields:
            name = field.getAttribute("name")

            if name == "bug_id":
                bug_id = str(field.childNodes[0].data)
            elif name == "summary":
                summary = str(field.childNodes[0].data)

                summary = summary.split(' ', 2)[2].strip()
            elif name == "description":
                if len(field.childNodes) > 0:
                    description = str(field.childNodes[0].data).strip()

        if bug_id != "397842":
            continue

        if bug_id != "" and int(bug_id) in used_bug_id:
            count += 1
            start_time = time.time()
            print(bug_id)
            text = summary.strip() + ". " + description.strip()
            print(text)
            print('---------------------------------------------')
            stack_trace = []

            start_index = 0

            for line in text.strip().split('\n'):
                line_res = [0] * len(line.strip().split())
                pattern = re.compile(
                    r'(((?P<exception_name>(\b[A-Za-z]+)?Exception)\sin\sthread)|(org\.)|(sun\.)|(java\.)|(at\s(\b[A-Za-z]+)(\.(\b[A-Za-z]+))*))([^\s]+)\.([^\s]+)((\((?P<file_name>(.+)\.(java|aj|main)(:\d+)*)\))|(\(Unknown Source\))|(\(Native Method\)))')
                matches = pattern.search(line)

                while matches is not None:
                    start_idx = matches.start()
                    end_idx = matches.end()
                    before_len = len(line[: start_idx].strip().split())
                    match_len = len(line[start_idx: end_idx].strip().split())
                    print(line[start_idx: end_idx])
                    line_res[before_len: before_len + match_len] = [1] * match_len

                    matches = pattern.search(line, end_idx)
                stack_trace += line_res

            print("==============================================")

            bug_id = int(bug_id)

            bug_token, st_labels, poly_labels = process_text(text, stack_trace)

            position_result = text_features.get_position(bug_token)

            co_occurrence_max_result, co_occurrence_mean_result, co_occurrence_median_result = text_features.co_occurrence(bug_token)

            token_code_df, token_tf_idf = text_features.get_tf_idf(bug_token, bug_id)

            token_span = text_features.get_br_span(bug_token)

            title = util.process_sent(summary.strip())

            title_freq = text_features.get_title_freq(bug_token, bug_id, title)

            br_freq = text_features.get_br_freq(bug_token)

            description = util.process_sent(description.strip())

            provenance = text_features.get_provenance(bug_id, bug_token, title, description)

            feedback_idf_result, feedback_tf_idf_result, feedback_max_freq_result, feedback_mean_freq_result, feedback_median_freq_result = bug_semantic.get_feedback_data(bug_id, bug_token)

            history_tf_idf_result, history_idf_result, history_max_freq_result, history_mean_freq_result, history_median_freq_result, history_key_time_result = bug_semantic.get_history_sim(bug_id, used_bug_id, bug_token, history_data, keyword_data, bug_target_code_data)

            title_token_sim, title_surround_token_sim = bug_semantic.get_title_sim(bug_token, bug_id, code_model, title, config.pooling_type)

            text_token_des, text_surround_token_des = bug_semantic.get_text_des(bug_token, ' '.join(bug_token), code_model, config.pooling_type)

            sim_code_token_df, sim_code_token_tf_idf, sim_code_token_max_freq, sim_code_token_mean_freq, sim_code_token_median_freq = code_semantic.text_sim(bug_token, bug_id, code_model, config.pooling_type)

            token_code_max_sim, token_code_mean_sim, token_code_median_sim, surround_code_max_sim, surround_code_mean_sim, surround_code_median_sim = code_semantic.code_sim(bug_token, bug_id, code_model, config.pooling_type)

            token_class_max_sim, token_class_mean_sim, token_class_median_sim, surround_class_max_sim, surround_class_mean_sim, surround_class_median_sim = code_semantic.class_sim(bug_token, bug_id, code_model, config.pooling_type)

            feedback_token_code_max_sim, feedback_token_code_mean_sim, feedback_token_code_median_sim, feedback_surround_code_max_sim, feedback_surround_code_mena_sim, feedback_surround_code_median_sim = code_semantic.feedback_sim(bug_token, bug_id, code_model, config.pooling_type)

            feedback_token_class_max_sim, feedback_token_class_mean_sim, feedback_token_class_median_sim, feedback_surround_class_max_sim, feedback_surround_class_mean_sim, feedback_surround_class_median_sim = code_semantic.feedback_classname_sim(bug_token, bug_id, code_model, config.pooling_type)

            end_time = time.time()
            print(end_time - start_time)
            result_list.append([end_time - start_time, len(position_result)])
            result_list.append([bug_id,
                                ' '.join(bug_token),
                                st_labels,
                                poly_labels,
                                position_result,
                                co_occurrence_max_result,
                                co_occurrence_mean_result,
                                co_occurrence_median_result,
                                token_code_df,
                                token_tf_idf,
                                token_span,
                                title_freq,
                                br_freq,
                                provenance,
                                feedback_idf_result,
                                feedback_tf_idf_result,
                                feedback_max_freq_result,
                                feedback_mean_freq_result,
                                feedback_median_freq_result,
                                history_tf_idf_result,
                                history_idf_result,
                                history_max_freq_result,
                                history_mean_freq_result,
                                history_median_freq_result,
                                history_key_time_result,
                                title_token_sim,
                                title_surround_token_sim,
                                text_token_des,
                                text_surround_token_des,
                                sim_code_token_df,
                                sim_code_token_tf_idf,
                                sim_code_token_max_freq,
                                sim_code_token_mean_freq,
                                sim_code_token_median_freq,
                                token_code_max_sim,
                                token_code_mean_sim,
                                token_code_median_sim,
                                surround_code_max_sim,
                                surround_code_mean_sim,
                                surround_code_median_sim,
                                token_class_max_sim,
                                token_class_mean_sim,
                                token_class_median_sim,
                                surround_class_max_sim,
                                surround_class_mean_sim,
                                surround_class_median_sim,
                                feedback_token_code_max_sim,
                                feedback_token_code_mean_sim,
                                feedback_token_code_median_sim,
                                feedback_surround_code_max_sim,
                                feedback_surround_code_mena_sim,
                                feedback_surround_code_median_sim,
                                feedback_token_class_max_sim,
                                feedback_token_class_mean_sim,
                                feedback_token_class_median_sim,
                                feedback_surround_class_max_sim,
                                feedback_surround_class_mean_sim,
                                feedback_surround_class_median_sim
                                ])
            print(result_list)
            break

    res = pd.DataFrame(result_list, columns=['bugId',
                                             'tokens',
                                             'is_stack_trace',
                                             'poly_num',
                                             "position",
                                             'co_occurrence_max',
                                             'co_occurrence_mean',
                                             'co_occurrence_median',
                                             'token_code_df',
                                             'token_tf_idf',
                                             'token_span',
                                             'title_freq',
                                             'br_freq',
                                             'provenance',
                                             'feedback_idf_result',
                                             'feedback_tf_idf_result',
                                             'feedback_max_freq_result',
                                             'feedback_mean_freq_result',
                                             'feedback_median_freq_result',
                                             'history_tf_idf_result',
                                             'history_idf_result',
                                             'history_max_freq_result',
                                             'history_mean_freq_result',
                                             'history_median_freq_result',
                                             'history_key_time_result',
                                             'title_token_sim',
                                             'title_surround_token_sim',
                                             'text_token_des',
                                             'text_surround_token_des',
                                             'sim_code_token_df',
                                             'sim_code_token_tf_idf',
                                             'sim_code_token_max_freq',
                                             'sim_code_token_mean_freq',
                                             'sim_code_token_median_freq',
                                             'token_code_max_sim',
                                             'token_code_mean_sim',
                                             'token_code_median_sim',
                                             'surround_code_max_sim',
                                             'surround_code_mean_sim',
                                             'surround_code_median_sim',
                                             'token_class_max_sim',
                                             'token_class_mean_sim',
                                             'token_class_median_sim',
                                             'surround_class_max_sim',
                                             'surround_class_mean_sim',
                                             'surround_class_median_sim',
                                             'feedback_token_code_max_sim',
                                             'feedback_token_code_mean_sim',
                                             'feedback_token_code_median_sim',
                                             'feedback_surround_code_max_sim',
                                             'feedback_surround_code_mena_sim',
                                             'feedback_surround_code_median_sim',
                                             'feedback_token_class_max_sim',
                                             'feedback_token_class_mean_sim',
                                             'feedback_token_class_median_sim',
                                             'feedback_surround_class_max_sim',
                                             'feedback_surround_class_mean_sim',
                                             'feedback_surround_class_median_sim'
                                             ])


def process_text(text, stack_trace):
    text = text.strip().split()
    st_labels = []
    tokens = []

    for index in range(len(text)):
        token = text[index]
        st_label = stack_trace[index]

        word_buf = []
        for character in token:
            if ('a' <= character <= 'z') or ('A' <= character <= 'Z'):
                word_buf.append(character)
                continue

            split_tokens = util.split_camel_case(word_buf)
            if (st_label == 0) or (st_label == 1 and len(split_tokens) == 1):
                st_labels += [0] * len(split_tokens)
            else:
                st_labels += [1] * len(split_tokens)
            tokens += split_tokens

            word_buf.clear()

        if len(word_buf) != 0:

            split_tokens = util.split_camel_case(word_buf)
            if (st_label == 0) or (st_label == 1 and len(split_tokens) == 1):
                st_labels += [0] * len(split_tokens)
            else:
                st_labels += [1] * len(split_tokens)
            tokens += split_tokens

    tokens, st_labels, poly_labels = preprocess(tokens, st_labels)

    return tokens, st_labels, poly_labels


def preprocess(tokens, st_labels):
    token_res = []
    st_label_res = []
    poly_res = []

    for index in range(len(tokens)):
        token = tokens[index].lower()
        syn_set = wordnet.synsets(token)
        token = ps.stem(token)
        if token not in ENG_STOP_WORDS_SET:
            token_res.append(token)
            st_label_res.append(st_labels[index])
            poly_res.append(len(syn_set))

    return token_res, st_label_res, poly_res


if __name__ == '__main__':
    is_stack_trace()
