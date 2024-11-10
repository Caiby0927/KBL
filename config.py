# project = "AspectJ"
# project = "Birt"
# project = "Eclipse_Platform_UI"
# project = "JDT"
# project = "SWT"
project = "Tomcat"
filepath_prefix = "Q:\\sata11-15601592758\\Exp\\query-reformulation\\dataset\\" + project
processed_filepath_prefix = "Q:\\sata11-15601592758\\Exp\\query-reformulation\\processed_dataset\\" + project

res_filepath_prefix = "E:\\query-reformulation\\test_dataset\\" + project

origin_dataset_filepath = "Q:\\sata11-15601592758\\Exp\\dataset\\the_dataset_of_six_open_source_Java_projects\\dataset\\" + project + ".csv"
origin_dataset_xml_filepath = "Q:\\sata11-15601592758\\Exp\\dataset\\the_dataset_of_six_open_source_Java_projects\\dataset\\" + project + ".xml"
embedding_model_path = processed_filepath_prefix + "\\embedding_models\\"

emb_size = 300
pooling_type = 1
train_text_filepath = processed_filepath_prefix + "\\train_bug.csv"
test_text_filepath = processed_filepath_prefix + "\\test_bug.csv"
top_n = 10

pseudo_feedback_filepath_prefix = "F:\\query-reformulation\\dataset\\" + project + "\\new\\data\\GA_result\\BugReport_"
code_dir_prefix = "E:\\query-reformulation\\dataset\\" + project + "\\data\\"
code_filepath_prefix = "F:\\query-reformulation\\dataset\\" + project + "\\new\\data\\"

win_size = 2
for_train = True
code_method_filepath_prefix = "Z:\\query-reformulation\\dataset\\" + project + "\\data\\"

regular_exp = r'((.*)?(.+)\.(.+)(\((.+)\.java:\d+\)|\(Unknown Source\)|\(Native Method\)))'

all_train_tokens_filepath = processed_filepath_prefix + "\\train_features_delete_zero.csv"
all_test_tokens_filepath = processed_filepath_prefix + "\\test_features_delete_zero.csv"

keyword_dup_constraint = 4
similarity = 0.6
share_keyword_file = processed_filepath_prefix + "\\BugCodeSharedKeyword.csv"

blizzard_filepath = filepath_prefix + "\\" + project.lower() + "_bugPredict.csv"
