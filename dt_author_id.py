#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn import tree
import sklearn.metrics as skm
sys.path.append("../tools/")
from email_preprocess import preprocess


#########################################################
### your code goes here ###
def val_dt(features_train, features_test, labels_train, labels_test, n):
    # t0 = time()
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    if n == 1:
        features_train = features_train[:len(features_train)/100]
        labels_train = labels_train[:len(labels_train)/100]
    train = clf.fit(features_train, labels_train)
    # print "training time:", round(time()-t0, 3), "s"
    # t1 = time()
    y_pred = train.predict(features_test)
    # print "training time:", round(time()-t1, 3), "s"
    acc=skm.accuracy_score(labels_test,y_pred)
    return clf, acc

#########################################################

print "3"
