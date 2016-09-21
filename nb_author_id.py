#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skm
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train1, features_test1, labels_train1, labels_test1 = preprocess()




#########################################################
### your code goes here ###
def val_nb(features_train, features_test, labels_train, labels_test):
    clf = GaussianNB()
    # t0 = time()
    train = clf.fit(features_train, labels_train)
    # print "training time:", round(time()-t0, 3), "s"
    # t1 = time()
    y_pred = train.predict(features_test)
    # print "training time:", round(time()-t1, 3), "s"
    acc=skm.accuracy_score(y_pred, labels_test)
    return clf, acc
#########################################################

print "1"