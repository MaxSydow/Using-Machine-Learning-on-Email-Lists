#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'

email_features_list = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

financial_features_list = [
    'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses',
    'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred',
    'salary', 'total_payments', 'total_stock_value']

# List of all features
features_list = ['poi'] + financial_features_list + email_features_list
#print features_list
     # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

keys = data_dict.keys()
print ('Number of data points: %d' %len(keys))
print ('Number of Features Per Person: %d' %len(data_dict[keys[0]].keys()))
num_poi = 0
num_real_salary = 0
for key, val in data_dict.items():
    if (val['poi'] == 1.0):
        num_poi = num_poi + 1
    if (val['salary'] != 'NaN'):
        num_real_salary = num_real_salary + 1

print ('Number of POIs In The Dataset: %d' % num_poi)
print ('Number of Non-POIs In The Dataset: %d' % (len(keys) - num_poi))


### Task 2: Remove outliers
#Identify and remove name outlier
#print keys
#data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

features_1 = ["salary", "bonus"]
data_1 = featureFormat(data_dict, features_1)
for point in data_1:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

# the outlying point in the plot clearly lies beyond bonus=8e8 and salary=2.5e7
for j in data_dict:
    if (data_dict[j]['bonus'] > 8000000 and data_dict[j]['bonus'] != 'NaN') and \
       (data_dict[j]['salary'] > 2500000 and data_dict[j]['salary'] != 'NaN'):
        print j
#Removing outlier
data_dict.pop('TOTAL', 0)


#Plotting without outlier
for point in data_1:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

# removing another outlier
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

#
def NaNAboveThreshold(data_dict, threshold):
    '''
    function to find element that have NaN values above a threshold percent
    '''
    keys = data_dict.keys()
    num_feat = len(data_dict[keys[0]].keys())
    dataKeys = []
    for name, item in data_dict.items():
        nan = 0
        for key, value in item.items():
            if (value == 'NaN'):
                nan = nan + 1

        nanPercent = nan / float(num_feat)
        if (nanPercent >= threshold):
            dataKeys.append(name)
    return dataKeys

# experimenting with higher thresholds
dataKeys = NaNAboveThreshold(data_dict, 0.85)
print dataKeys
for name in dataKeys:
    print name, " " , data_dict[name]
# LOCKHART EUGENE E has all NaN values, this name should be removed
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)

def prop_email(poi_emails, all_emails):
    """
    return the fraction of messages to/from that person that are from/to a POI
    """
    if poi_emails == 'NaN' or all_emails == 'NaN' or poi_emails == 0 or all_emails == 0:
        fraction = 0.
    else:
        fraction = float(poi_emails) / float(all_emails)
    return fraction

for name in data_dict:
    data_name = data_dict[name]
    data_name["prop_email_from_poi"] = \
        prop_email(data_name["from_poi_to_this_person"], data_name["to_messages"])
    data_name["prop_email_to_poi"] = \
        prop_email(data_name["from_this_person_to_poi"], data_name["from_messages"])
features_list = features_list + ['prop_email_to_poi'] + ['prop_email_from_poi']

# function for k-best-features and scores
def findKbestFeatures(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    print "Features and scores: ", sorted_pairs
    k_best_features = dict(sorted_pairs[:k])
    #print "{0} best features: {1}, {2}\n".format(k, k_best_features.keys(), scores)
    return k_best_features


# Get K-best features
num_features = 5
features_list.remove('email_address') # remove string values for k-best

selectedBestFeatures = findKbestFeatures(data_dict, features_list, num_features)
selectedFeatures = selectedBestFeatures.keys()
print "Highest scoring features:  ", selectedFeatures

features_used = ['poi', 'exercised_stock_options']
#features_used = ['poi', 'bonus', 'exercised_stock_options', 'prop_email_to_poi', 'salary', 'total_stock_value']
print features_used
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_used, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scaling features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# split features into training and test data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state = 42)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Classifiers:

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

##clf = GaussianNB()
#clf = SVC()
#clf = tree.DecisionTreeClassifier()
#clf = AdaBoostClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# use GridSearch to optimize parameters
from sklearn.grid_search import GridSearchCV
parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
              'C': [1, 10, 100, 1000, 10000],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

clf = GridSearchCV(SVC(), parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
# ratio of true positives to all positives
precision = precision_score(labels_test, pred)
# ratio of tru positives to true positives and false negatives
recall = recall_score(labels_test, pred)
# cross validation scores
xvscore = clf.score(features_test, labels_test)

print "Accuracy: ", accuracy
print 'Precision :', precision
print 'Recall :', recall
#print 'Cross Validation Score: ', xvscore

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf, features, labels, cv = 3)
print 'Validation Scores: ', scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clfNB,  my_dataset, features_list)
#dump_classifier_and_data(clfSVM,  my_dataset, features_list)
#dump_classifier_and_data(clfDT,  my_dataset, features_list)
dump_classifier_and_data(clf,  my_dataset, features_list)
