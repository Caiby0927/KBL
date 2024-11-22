# KBL: A Golden Keywords-based Query Reformulation Approach for Bug Localization

## Getting Started
Clone the repository by using the following command.

`git clone https://github.com/Caiby0927/KBL.git`

If you don't have git, please install git first using following commands.

$ sudo apt-get update
$ sudo apt-get install git

## Install Python
We use python 3.8

$ sudo add-apt-repository ppa:fkrull/deadsnakes

$ sudo apt-get update

$ sudo apt-get install python3.8 python

$ sudo apt-get install python-pip

## Install python libraries
We have 5 dependencies below:

```
lightgbm >= 3.3.2
pandas >= 1.1.4
numpy >= 1.13.3
scipy >= 0.19.1
nltk >= 3.7
```

## Execution
Step1: Download the bug reports from the bug tracking systems of Eclipse foundations
The bug reports of Eclipse foundation could be downloaded from https://bugs.eclipse.org/

Step2: Obtain the preliminary keywords for bug reports
After getting the bug report, you can use the code in the [GA](https://github.com/Caiby0927/Genetic_Algorithm) repository (also written by us) to get the preliminary keywords corresponding to each bug report.

Step3: Refine the preliminary keywords
With the preliminary keywords, we can use the keyword_refinement.java, which is also in the [GA](https://github.com/Caiby0927/Genetic_Algorithm) repository, to refine them. After refinement, we can get the golden keywords for each bug report.

Step4: Extract features for terms in the bug report
To build the keywords classifier, we first need to extract features from each term in the bug report. We can use the scripts call bug_semantic.py, code_semantic.py and camel_case.py to extract all 61 features for the term.

Step5: Build the keywords classifier
Before training the classifier, we need to label the terms which are used for training and testing. Here, we use the golden keywords to label each term. Then, we can use the script train.py to build the keywords classifier.

Step6: Predict keywords for new bug report
After building the classifier, we can input the new bug report and use the trained classifier to predict the keywords of the bug report with the script call eval.py.
