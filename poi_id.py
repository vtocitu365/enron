#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from nb_author_id import val_nb
from svm_author_id import val_svm
from dt_author_id import val_dt
from finance_regression import val_reg
from k_means_cluster import val_knn
from tester import test_classifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
import sklearn.feature_selection as skf
import sklearn.metrics as skm

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock',
                 'shared_receipt_with_poi', 'expenses', 'from_messages', 'other',
                 'long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Get rid of from_message < 6000
# Get rid of to_message > 10000
# Ger rid of NaN
for x in data_dict:
    for y in range(0, len(features_list)):
        if data_dict[x][features_list[y]] == 'NaN': data_dict[x][features_list[y]] = {}

data_dict.pop("TOTAL",0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for x in data_dict :
    print data_dict[x]['bonus'], "hey"
    print data_dict[x]['total_payments']
    data_dict[x]['importance_of_bonus'] = data_dict[x]['bonus']/data_dict[x]['total_payments']


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
test1 = 0.3
numValues = 0

def val_selectKBest(features, labels):
    clf = skf.SelectKBest(skf.f_classif)
    fitting = clf.fit(features,labels)
    feature = fitting.transform(features)
    kbest_scores = clf.scores_
    return feature, kbest_scores

features, kbest_scores = val_selectKBest(features, labels)
print kbest_scores

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test1, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
[clf_nb, accuracy_nb] = val_nb(features_train, features_test, labels_train, labels_test)
[clf_svm, accuracy_svm] = val_svm(features_train, features_test, labels_train, labels_test, "rbf",1000, numValues)
[clf_reg, accuracy_reg]= val_reg(features_train, features_test, labels_train, labels_test)
[clf_dt, accuracy_dt]=val_dt(features_train, features_test, labels_train, labels_test, numValues)
[clf_knn, pred1, acc5] = val_knn(features,labels,clusters=2)
accuracy_total = [accuracy_nb, accuracy_svm, accuracy_reg, accuracy_dt, acc5]
clf_total = [clf_nb, clf_svm, clf_reg, clf_dt, clf_knn]
print accuracy_svm

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

###############################################################################
# Train a SVM classification model
def val_grid(enron_method, feature_train, target_train, param_grid):
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf = GridSearchCV(enron_method, param_grid)
    clf = clf.fit(feature_train, target_train)
    accuracy_grid = clf.score(feature_train, target_train)
    print "Best estimator found by grid search:"
    print clf.best_estimator_
    return clf.best_estimator_, accuracy_grid


###############################################################################

enron_method_svm = SVR(kernel='rbf')
param_grid_svm = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
[clf_grid, acc5] = val_grid(enron_method_svm, features_train, labels_train, param_grid_svm)

print acc5

[accuracy, precision, recall, f1, f2] = test_classifier(clf_svm, pd.DataFrame(data_dict), features_list, folds = 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf_reg, my_dataset, features_list)
