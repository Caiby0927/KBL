import collections
import random

import joblib
import jpype
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
import construct_data
import eval


def read_training_data(random_state=None, feature_list=None, parameters=None, gbm=None):
	df_test = pd.read_csv(config.all_test_tokens_filepath)

	x_train_df, y_train = construct_data.other_balance()

	y_test = df_test['label'].values
	x_test_df = df_test.drop([
		'bugId',
		'token',
		'label'
	], axis=1)

	if feature_list is not None:
		x_train_df = x_train_df[feature_list]
		x_test_df = x_test_df[feature_list]

	x_train = x_train_df.values
	x_test = x_test_df.values

	lgb_train = lgb.Dataset(x_train, y_train)

	params = {
		'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'binary',
		'learning_rate': 0.03,
		'feature_fraction': 0.7,
		'feature_fraction_seed': 1945,
		'num_leaves': 104,
		'max_depth': 10,
		'metric': ['rmse'],
		'min_child_weight': 0.001,
		'min_child_samples': 21,
		'reg_alpha': 0,
		'reg_lambda': 0.001,
		'verbose': -1
	}

	print('start training...')

	gbm = lgb.train(params, lgb_train, 10000)

	print('saving model...')
	joblib.dump(gbm, config.filepath_prefix + "\\" + model_name + "_model.pkl")

	print('start predicting...')
	y_pred = gbm.predict(x_test)
	y = []
	with open(config.res_filepath_prefix + "\\predict_result_" + model_name + ".txt", 'w+') as f:
		for pred in y_pred:
			f.write(str(pred) + "\n")
			if pred > 0.5:
				y.append(1)
			else:
				y.append(0)
		f.flush()

	print(confusion_matrix(y_true=y_test, y_pred=y))

	tp_val = 0
	fp_val = 0
	tn_val = 0
	fn_val = 0
	df_test = df_test[['bugId', 'token', 'label']]
	df_test['pred'] = y
	test_bug_ids = set(df_test['bugId'].tolist())
	for test_bug in test_bug_ids:
		bug_data = df_test[df_test['bugId'] == test_bug]
		tokens = set(bug_data['token'].tolist())
		for token in tokens:
			token_data = bug_data[bug_data['token'] == token]
			labels = sorted(token_data['label'].tolist())
			preds = sorted(token_data['pred'].tolist())
			for i in range(len(labels)):
				if labels[i] == preds[i] and labels[i] == 1:
					tp_val += 1
				elif labels[i] == preds[i] and labels[i] == 0:
					tn_val += 1
				elif labels[i] == 1:
					fn_val += 1
				else:
					fp_val += 1

	df_test['ml_pred'] = y_pred
	df_test.to_csv(
		config.res_filepath_prefix + "\\test_bug_pred_result_" + model_name + ".csv",
		columns=['bugId', 'token', 'label', 'ml_pred'],
		index=False
	)
	print('finish predicting...')


def run_retrieval():
	java_class = jpype.JClass('org.query.util.EvalDataProcess')
	instance = java_class()
	instance.main([
		config.project,
		model_name,
		predict_res_file,
		rank_res_file
	])


def tune():
	with open(config.filepath_prefix + "\\random_sample_results_1_balance_by_br.txt", 'a+') as f:
		max_val = 100000
		for i in range(50):

			random_state = random.randint(100, max_val)
			print(random_state)
			read_training_data(random_state)

			print("start evaluating...")
			eval.evaluate(model_name)
			run_retrieval()
			acc_res, map_res, mrr_res = eval.compute_result(model_name)
			print("finish evaluating...")
			f.write("random state: " + str(
				random_state) + "\taccuracy: " + acc_res + "\tmap: " + map_res + "\tmrr: " + mrr_res + "\n")
			f.flush()
			break


def tune_param():
	parameters = {
		'reg_alpha': [0, 0.001, 0.002],
		'reg_lambda': [0, 1, 2]
	}

	gbm = lgb.LGBMClassifier(objective='binary', num_leaves=90, learning_rate=0.1, max_depth=10, feature_fraction=0.7,
							 bagging_fraction=1.0, bagging_freq=1, metric='rmse', min_child_samples=20,
							 min_child_weight=0.001,
							 reg_alpha=0.002, reg_lambda=0, num_iterations=10000)

	read_training_data(parameters=parameters, gbm=gbm)


def check_every_single_feature(feature_list=None):
	full_feature_f1 = read_training_data(feature_list=feature_list)
	print("full features f1 score {}".format(full_feature_f1))
	tmp_features = []
	for feature in feature_list:
		tmp_features.append(feature)

	for i in range(len(feature_list)):
		print("checking the feature {}".format(feature_list[i]))
		tmp_features.remove(feature_list[i])
		print(len(tmp_features))
		feature_f1 = read_training_data(feature_list=tmp_features)
		print("without {} features f1 score {}".format(feature_list[i], feature_f1))
		if feature_f1 > full_feature_f1:
			print("{} feature is useful".format(feature_list[i]))
		else:
			print("{} feature is not good".format(feature_list[i]))
		feature_importance = feature_f1 - full_feature_f1
		print("feature importance is {}".format(feature_importance))

		eval.evaluate(model_name)
		run_retrieval()
		eval.compute_result(model_name)
		tmp_features.append(feature_list[i])

	final_feature_f1 = read_training_data(feature_list=tmp_features)
	print("final features f1 score {}".format(final_feature_f1))
	print(tmp_features)


jar_path = "E:\\query-reformulation\\query_reformulation\\out\\artifacts\\eval_tomcat_jar\\query_reformulation.jar"
jvm_path = "C:\\Users\\HS\\.jdks\\corretto-15.0.2\\bin\\server\\jvm.dll"
jpype.startJVM(jvm_path, "-ea", "-Djava.class.path=%s" % (jar_path))
#
model_name = "lightgbm_ROS"
predict_res_file = config.res_filepath_prefix + "\\predict_result_" + model_name + ".csv"
rank_res_file = config.res_filepath_prefix + "\\predict_rank_result_" + model_name + ".csv"

read_training_data()
eval.evaluate(model_name)
run_retrieval()
eval.compute_result(
	model_name,
	res_file="E:\\query-reformulation\\processed_dataset\\" + config.project + "\\GA_total_result.csv"
)
jpype.shutdownJVM()