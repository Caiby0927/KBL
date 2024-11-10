import pandas as pd

import config

from xml.dom.minidom import parse
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import spacy

import data

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

ps = PorterStemmer()
ENG_STOP_WORDS_SET = set()
for word in stopwords.words('english'):
    word = word.lower().strip()
    word = ps.stem(word)
    ENG_STOP_WORDS_SET.add(word)


class CamelCase(object):
    def __init__(self):
        self.project = config.project
        self.filepath = "F:\\dataset\\the_dataset_of_six_open_source_Java_projects\\dataset\\" + self.project + ".xml"
        self.bug_list = []
        self.nlp = spacy.load('en_core_web_sm')

    def run(self):
        self.read_xml()
        res = pd.DataFrame(self.bug_list, columns=['bugId',
                                                   'text',
                                                   'camel_case',
                                                   'is_class_name',
                                                   'pos',
                                                   'dep',
                                                   'is_method_name'])

        res.to_csv(config.filepath_prefix + "\\camel_case.csv", index=False)

    def read_xml(self):
        dom_tree = parse(self.filepath)
        root_node = dom_tree.documentElement

        bugs = root_node.getElementsByTagName("table")
        self.bug_list = []
        used_bug_id = data.get_used_br()
        for bug in bugs:
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

            if bug_id != "" and int(bug_id) in used_bug_id:
                text = summary.strip() + ". " + description.strip()

                class_names = data.get_class_name(bug_id)
                method_names = data.get_method_name(bug_id)

                bug_label_list = []
                bug_is_class_label_list = []
                bug_pos_label_list = []
                bug_dep_label_list = []
                bug_is_method_label_list = []

                sent_list = sent_tokenize(text)
                for sent in sent_list:
                    sent_analysis = self.nlp(sent)
                    sent_pos_res = []
                    sent_dep_res = []
                    for token in sent_analysis:

                        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                            pos = 1
                        elif token.pos_ == 'VERB':
                            pos = 2
                        elif token.pos_ == 'ADJ':
                            pos = 3
                        else:
                            pos = 0
                        sent_pos_res.append([token.text, pos])

                        token_dep = token.dep_
                        if token_dep == 'ROOT':
                            dep = 1
                        elif token_dep == 'nsubj':
                            dep = 2
                        elif token_dep == 'doubj':
                            dep = 3
                        elif token_dep == 'prep':
                            dep = 4
                        elif token_dep == 'probj':
                            dep = 5
                        elif token_dep == 'cc':
                            dep = 6
                        elif token_dep == 'compound':
                            dep = 7
                        elif token_dep == 'advmod':
                            dep = 8
                        elif token_dep == 'det':
                            dep = 9
                        elif token_dep == 'amod':
                            dep = 10
                        else:
                            dep = 0
                        sent_dep_res.append([token.text, dep])

                    bug_token_list, label_list, is_class_label_list, pos_label_list, dep_label_list, is_method_label_list = self.process_text(
                        sent,
                        class_names,
                        sent_pos_res,
                        sent_dep_res,
                        method_names
                    )

                    bug_token_list, label_list, is_class_label_list, pos_label_list, dep_label_list, is_method_label_list = self.preprocess(
                        bug_token_list,
                        label_list,
                        is_class_label_list,
                        pos_label_list,
                        dep_label_list,
                        is_method_label_list
                    )

                    bug_label_list += label_list
                    bug_is_class_label_list += is_class_label_list
                    bug_pos_label_list += pos_label_list
                    bug_dep_label_list += dep_label_list
                    bug_is_method_label_list += is_method_label_list

                self.bug_list.append([bug_id,
                                      text,
                                      bug_label_list,
                                      bug_is_class_label_list,
                                      bug_pos_label_list,
                                      bug_dep_label_list,
                                      bug_is_method_label_list])

    def process_text(self, text, class_names, pos, dep, method_names):
        bug_token_list = []
        bug_label_list = []
        bug_is_class_label_list = []
        bug_pos_label_list = []
        bug_dep_label_list = []
        bug_is_method_label_list = []

        word_buf = []
        idx = 0
        space_pre = False
        for character in text:
            if ('a' <= character <= 'z') or ('A' <= character <= 'Z'):
                space_pre = False
                word_buf.append(character)
                continue

            pos_label = pos[idx][1]
            dep_label = dep[idx][1]

            tokens, labels, is_class_label, pos_labels, dep_labels, is_method_label = self.split_camel_case(word_buf,
                                                                                                            pos_label,
                                                                                                            dep_label,
                                                                                                            class_names,
                                                                                                            method_names)
            bug_token_list += tokens
            bug_label_list += labels
            bug_is_class_label_list += is_class_label
            bug_pos_label_list += pos_labels
            bug_dep_label_list += dep_labels
            bug_is_method_label_list += is_method_label

            if not space_pre and (character == ' ' or character == '\n' or character == '\t'):
                space_pre = True
                idx += 1
            word_buf.clear()

        pos_label = pos[idx][1]
        dep_label = dep[idx][1]
        tokens, labels, is_class_label, pos_labels, dep_labels, is_method_label = self.split_camel_case(word_buf,
                                                                                                        pos_label,
                                                                                                        dep_label,
                                                                                                        class_names,
                                                                                                        method_names)
        bug_token_list += tokens
        bug_label_list += labels
        bug_is_class_label_list += is_class_label
        bug_pos_label_list += pos_labels
        bug_dep_label_list += dep_labels
        bug_is_method_label_list += is_method_label

        return bug_token_list, bug_label_list, bug_is_class_label_list, bug_pos_label_list, bug_dep_label_list, bug_is_method_label_list

    def split_camel_case(self, word_buf, pos, dep, class_names=None, method_names=None):
        if class_names is None:
            class_names = []
        if method_names is None:
            method_names = []

        tokens = []
        labels = []
        is_class_label = []
        pos_label = []
        dep_label = []
        is_method_label = []

        length = len(word_buf)
        if length != 0:
            k = 0
            i = 0
            j = 1
            meet_camel_case = False
            class_label = 0
            method_label = 0

            complete_word = ''.join(word_buf)
            if complete_word in class_names:
                class_label = 1
            if complete_word in method_names:
                method_label = 1

            while i < length - 1:
                first = word_buf[i]
                second = word_buf[j]
                if ('A' <= first <= 'Z') and ('a' <= second <= 'z'):
                    token = ''.join(word_buf[k: i]).strip()
                    if token != "" and len(token) > 0:
                        meet_camel_case = True
                        tokens.append(token)
                        labels.append(1)
                        is_class_label.append(class_label)
                        pos_label.append(pos)
                        dep_label.append(dep)
                        is_method_label.append(method_label)
                    k = i
                    i += 1
                    j += 1
                    continue
                if ('a' <= first <= 'z') and ('A' <= second <= 'Z'):
                    token = ''.join(word_buf[k: j]).strip()
                    if token != "" and len(token) > 0:
                        meet_camel_case = True
                        tokens.append(token)
                        labels.append(1)
                        is_class_label.append(class_label)
                        pos_label.append(pos)
                        dep_label.append(dep)
                        is_method_label.append(method_label)
                    k = j
                i += 1
                j += 1

            if k < length:
                token = ''.join(word_buf[k:])
                if token.strip() != "" and len(token) > 0:
                    tokens.append(token)
                    if meet_camel_case:
                        labels.append(1)
                    else:
                        labels.append(0)
                    is_class_label.append(class_label)
                    pos_label.append(pos)
                    dep_label.append(dep)
                    is_method_label.append(method_label)

        return tokens, labels, is_class_label, pos_label, dep_label, is_method_label

    def preprocess(self, bug_token_list, bug_label_list, bug_is_class_label_list, bug_pos_label_list,
                   bug_dep_label_list, bug_is_method_label_list):
        token_res_list = []
        label_res_list = []
        is_class_res_list = []
        pos_label_res_list = []
        dep_label_res_list = []
        is_method_res_list = []

        for index in range(len(bug_token_list)):
            token = ps.stem(bug_token_list[index].lower())
            if token not in ENG_STOP_WORDS_SET:
                token_res_list.append(token)
                label_res_list.append(bug_label_list[index])
                is_class_res_list.append(bug_is_class_label_list[index])
                pos_label_res_list.append(bug_pos_label_list[index])
                dep_label_res_list.append(bug_dep_label_list[index])
                is_method_res_list.append(bug_is_method_label_list[index])

        return token_res_list, label_res_list, is_class_res_list, pos_label_res_list, dep_label_res_list, is_method_res_list


if __name__ == '__main__':
    camel_case = CamelCase()
    camel_case.run()
