#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2", f3_name = "feature 3"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)
dd1=[]
for x in data_dict : dd1. append(data_dict[x]['exercised_stock_options'])
dd1 = filter(lambda a: a!='NaN' , dd1)
dd2=[]
for x in data_dict : dd2. append(data_dict[x]['salary'])
dd2 = filter(lambda a: a!='NaN' , dd2)
dd = [max(dd1), min(dd1), max(dd2), min(dd2)]
### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"

poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


'''
### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()
'''
### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
def val_knn(finance_features,poi,clusters):
    minmaxscaler = MinMaxScaler()
    features_rescaled = minmaxscaler.fit_transform(finance_features)
    clf = KMeans(n_clusters=clusters)
    pred = clf.fit_predict(finance_features)
    acc=skm.accuracy_score(poi,pred)
    return clf, pred, acc

[clf, pred, acc] = val_knn(finance_features,poi,clusters=2)
print acc
'''
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2, f3_name=feature_3)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
'''
