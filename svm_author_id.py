#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import sklearn.svm as skv
import sklearn.metrics as skm
sys.path.append("../tools/")


#########################################################
### your code goes here ###
def val_svm(features_train, features_test, labels_train, labels_test, Kkernel, dC, n):
    # t0 = time()
    clf = skv.SVC(C=dC, kernel=Kkernel)
    if n == 1:
        features_train = features_train[:len(features_train)/100]
        labels_train = labels_train[:len(labels_train)/100]
    train = clf.fit(features_train, labels_train)
    # print "training time:", round(time()-t0, 3), "s"
    #t1 = time()
    y_pred = train.predict(features_test)
    # print "training time:", round(time()-t1, 3), "s"
    acc=skm.accuracy_score(labels_test,y_pred)
    return clf, acc
#########################################################
print "2"