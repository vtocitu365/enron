#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )
from sklearn.cross_validation import train_test_split




### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.


def val_reg(feature_train, feature_test, target_train, target_test):
    clf = LinearRegression()
    reg = clf.fit(feature_train, target_train)
    print clf.coef_,clf.intercept_
    y_pred = reg.predict(feature_test)
    acc1=clf.score(feature_train,target_train)
    acc2=clf.score(feature_test,target_test)
    return clf, acc1
print "4"