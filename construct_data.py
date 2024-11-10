import collections

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
import config
import data
import util


def split_features(keyword_data, train_bug, test_bug):
	filepath = config.res_filepath_prefix + "\\merge_features.csv"
	feature_data = pd.read_csv(filepath)

	feature_data = feature_data.T.drop_duplicates().T
	print(feature_data.columns)

	train_features = []
	test_features = []

	for index, row in feature_data.iterrows():
		features = []

		bug_id = row['bugId']
		if int(bug_id) not in train_bug and int(bug_id) not in test_bug and str(bug_id) not in train_bug and str(
				bug_id) not in test_bug:
			continue
		tokens = str(row['tokens']).strip().split()
		features.append(tokens)
		token_num = len(tokens)

		label_list = get_token_label(bug_id, keyword_data, tokens)

		# # is_from_stack trace
		features.append(str(row['is_stack_trace']).strip()[1: -1].split(','))
		# # term_meaning_variety
		features.append(str(row['poly_num']).strip()[1: -1].split(','))
		# # term_position
		features.append(str(row['position']).strip()[1: -1].split(','))
		# # BR_term_co-occurrence
		features.append(str(row['co_occurrence_max']).strip()[1: -1].split(','))
		features.append(str(row['co_occurrence_mean']).strip()[1: -1].split(','))
		# features.append(str(row['co_occurrence_median']).strip()[1: -1].split(','))
		# term_df
		features.append(str(row['token_code_df']).strip()[1: -1].split(','))
		# term_tf_idf
		features.append(str(row['token_tf_idf']).strip()[1: -1].split(','))
		# term_span
		features.append(str(row['token_span']).strip()[1: -1].split(','))
		# # term_title_tf
		features.append(str(row['title_freq']).strip()[1: -1].split(','))
		# # term_br_tf
		features.append(str(row['br_freq']).strip()[1: -1].split(','))
		# # term_source
		features.append(str(row['provenance']).strip()[1: -1].split(','))
		# feedback_term_statistic
		features.append(str(row['feedback_idf_result']).strip()[1: -1].split(','))
		features.append(str(row['feedback_tf_idf_result']).strip()[1: -1].split(','))
		features.append(str(row['feedback_max_freq_result']).strip()[1: -1].split(','))
		features.append(str(row['feedback_mean_freq_result']).strip()[1: -1].split(','))
		features.append(str(row['feedback_median_freq_result']).strip()[1: -1].split(','))
		# similar_br_term_statistic
		features.append(str(row['history_tf_idf_result']).strip()[1: -1].split(','))
		features.append(str(row['history_idf_result']).strip()[1: -1].split(','))
		features.append(str(row['history_max_freq_result']).strip()[1: -1].split(','))
		features.append(str(row['history_mean_freq_result']).strip()[1: -1].split(','))
		features.append(str(row['history_median_freq_result']).strip()[1: -1].split(','))
		# similar_br_term_importance
		features.append(str(row['history_key_time_result']).strip()[1: -1].split(','))
		# # term_title_similarity
		features.append(str(row['title_token_sim']).strip()[1: -1].split(','))
		features.append(str(row['title_surround_token_sim']).strip()[1: -1].split(','))
		# # term_title_importance
		features.append(str(row['text_token_des']).strip()[1: -1].split(','))
		features.append(str(row['text_surround_token_des']).strip()[1: -1].split(','))
		# similar_code_term_statistic
		features.append(str(row['sim_code_token_df']).strip()[1: -1].split(','))
		features.append(str(row['sim_code_token_tf_idf']).strip()[1: -1].split(','))
		features.append(str(row['sim_code_token_max_freq']).strip()[1: -1].split(','))
		features.append(str(row['sim_code_token_mean_freq']).strip()[1: -1].split(','))
		features.append(str(row['sim_code_token_median_freq']).strip()[1: -1].split(','))
		# term_code_similarity
		features.append(str(row['token_code_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['token_code_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['token_code_median_sim']).strip()[1: -1].split(','))
		features.append(str(row['surround_code_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['surround_code_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['surround_code_median_sim']).strip()[1: -1].split(','))
		# term_classname_similarity
		features.append(str(row['token_class_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['token_class_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['token_class_median_sim']).strip()[1: -1].split(','))
		features.append(str(row['surround_class_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['surround_class_mean_sim']).strip()[1: -1].split(','))
		# features.append(str(row['surround_class_median_sim']).strip()[1: -1].split(','))
		# term_feedback_similarity
		features.append(str(row['feedback_token_code_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_token_code_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_token_code_median_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_code_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_code_mena_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_code_median_sim']).strip()[1: -1].split(','))
		# term_feedback_classname_similarity
		features.append(str(row['feedback_token_class_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_token_class_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_token_class_median_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_class_max_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_class_mean_sim']).strip()[1: -1].split(','))
		features.append(str(row['feedback_surround_class_median_sim']).strip()[1: -1].split(','))
		# is_from_camel_case
		features.append(str(row['camel_case']).strip()[1: -1].split(','))
		# is_from_class_name
		features.append(str(row['is_class_name']).strip()[1: -1].split(','))
		# pos_tag
		features.append(str(row['pos']).strip()[1: -1].split(','))
		# term_syntactic_dependency_relationship
		features.append(str(row['dep']).strip()[1: -1].split(','))
		# is_from_method_name
		features.append(str(row['is_method_name']).strip()[1: -1].split(','))
		features.append(label_list)

		token_features = []
		for col_idx in range(token_num):
			token_feature = [bug_id]
			for row_idx in range(len(features)):
				token_feature.append(features[row_idx][col_idx])
			# print(len(token_feature))
			token_features.append(token_feature)

		if bug_id in train_bug:
			train_features += token_features
		else:
			test_features += token_features
	# print(len(token_features))

	print(len(train_features))
	columns = ['bugId',
			   'token',
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
			   'feedback_surround_code_mean_sim',
			   'feedback_surround_code_median_sim',
			   'feedback_token_class_max_sim',
			   'feedback_token_class_mean_sim',
			   'feedback_token_class_median_sim',
			   'feedback_surround_class_max_sim',
			   'feedback_surround_class_mean_sim',
			   'feedback_surround_class_median_sim',
			   'camel_case',
			   'is_class_name',
			   'pos',
			   'dep',
			   'is_method_name',
			   'label']
	res = pd.DataFrame(train_features, columns=columns)
	res.to_csv(config.res_filepath_prefix + "\\train_features.csv", index=False)
	res = pd.DataFrame(test_features, columns=columns)
	res.to_csv(config.res_filepath_prefix + "\\test_features.csv", index=False)


def get_token_label(bug_id, keyword_data, token_list):
	br_keyword_data = keyword_data[keyword_data['bugId'] == bug_id]
	keywords = str(br_keyword_data['keywords'].values[0]).strip().split()

	label_list = [0] * len(token_list)

	keyword_counter = dict(collections.Counter(keywords))

	for key, value in keyword_counter.items():
		token_idx = util.get_index(key, token_list)

		if len(token_idx) == 0:
			continue
		for idx in range(value):
			try:
				label_list[token_idx[idx]] = 1
			except:
				break

	return label_list


def data_balance(sample_random_state=None, file_name=None, do_balance=True):
	sample_random_state = 1391

	if file_name is None:
		train_tokens = pd.read_csv(config.all_train_tokens_filepath)
	else:
		train_tokens = pd.read_csv(file_name)

	if not do_balance:
		print("no balance")
		return 0, train_tokens

	train_true_data = train_tokens[train_tokens['label'] == 1]
	n = len(train_true_data) * 2
	train_false_data = train_tokens[train_tokens['label'] == 0]
	class_weight = (len(train_false_data) / len(train_true_data))

	train_false_data = train_false_data.sample(n, random_state=sample_random_state)

	balanced_data = pd.concat([train_true_data, train_false_data])
	balanced_data = balanced_data.sample(frac=1, random_state=sample_random_state)

	return class_weight, balanced_data


def other_balance():

	train_tokens = pd.read_csv(config.all_train_tokens_filepath)
	# balance = TomekLinks()
	# balance = NeighbourhoodCleaningRule()
	balance = RandomOverSampler()

	y_train = train_tokens['label'].values
	print(collections.Counter(y_train))
	x_train_df = train_tokens.drop(['bugId', 'token', 'label'], axis=1)

	x_train_df, y_train = balance.fit_resample(x_train_df, y_train)
	return x_train_df, y_train


def data_scalar(data_df):
	scalar = MinMaxScaler()
	columns = list(data_df.columns)
	columns = [x for x in columns if x != "bugId" and x != "token" and x != "label"]
	data_df[columns] = scalar.fit_transform(data_df[columns])
	# print(data_df)
	# print(data_df.columns)
	return data_df


if __name__ == '__main__':
	data_balance()
