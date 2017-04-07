#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV

### Import helper functions
from helper_functions import plotting_salary_expenses, dataset_info,\
    remove_outlier, ratio, find_empty

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
                'bonus',
                'long_term_incentive',
                'deferred_income',
                'deferral_payments',
                'loan_advances',
                'other',
                'expenses',
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#my_dataset = data_dict

# Get info about this dataset
dataset_info(data_dict)
### Task 2: Remove outliers
# I will plot some of the features to check for outliers and see how the data distributes on the plot
plotting_salary_expenses(data_dict, 'salary', 'expenses')
plotting_salary_expenses(data_dict, 'salary', 'from_poi_to_this_person')
plotting_salary_expenses(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
plotting_salary_expenses(data_dict, 'salary', 'from_this_person_to_poi')
plotting_salary_expenses(data_dict, 'shared_receipt_with_poi', 'to_messages')
# From these plots I can see only one major outlier. So I remove it:
outliers = ['TOTAL']
remove_outlier(data_dict, outliers)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
# Create ratio from person to poi and vice versa
for name in my_dataset:
    data_point = my_dataset[name]
    poi_to = data_point["from_poi_to_this_person"]
    poi_from = data_point["from_this_person_to_poi"]
    ratio_from_poi = ratio(poi_to, poi_from)
    data_point["ratio_of_poi_emails"] = ratio_from_poi

# Plotting the new feature
plotting_salary_expenses(data_dict, 'salary', 'ratio_of_poi_emails')
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Counting NAN
find_empty(data_dict)

# Selecting features using k-best
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(k=10)
# selected_features are the features selected by SelectKbest
selected_features = kbest.fit_transform(features, labels)
features_selected =[features_list[i+1] for i in kbest.get_support(indices=True)]
for f in features_selected[0:]:
	print f, "score is: ", kbest.scores_[features_selected[0:].index(f)]

# Scaling the features
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Accuracy: 0.76800	Precision: 0.19544	Recall: 0.16300	F1: 0.17775	F2: 0.16860
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

#Accuracy: 0.75677	Precision: 0.24428	Recall: 0.27750	F1: 0.25983	F2: 0.27015
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier()

#Accuracy: 0.81231	Precision: 0.00673	Recall: 0.00150	F1: 0.00245	F2: 0.00178
from sklearn.neighbors import KNeighborsClassifier
clf3 = KNeighborsClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, cross_val_score

# Plot features in to train and test.
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

skb = SelectKBest(k = 10)
# using the same pipeline:
pipe = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("Naive bayes", GaussianNB())])

# define the parameter grid for SelectKBest,
# using the name from the pipeline followed by 2 underscores:
parameters = {'SKB__k': range(1, 14)}

# Use the pipeline in GridSearchCV, with the parameter 'grid'
# using 'f1' as the scoring metric (as it is the weighted average
# of precision and recall):
gs = GridSearchCV(pipe, param_grid = parameters, scoring = 'f1')

# fit GridSearchCV:
gs.fit(features_train, labels_train)

# extract the best algorithm:
clf = gs.best_estimator_

print 'best algorithm is: '
print clf

# create an instance of 'StratifiedShuffleSplit',
# in this case '100' refers to the number of folds
# that is, the number of test/train splits
sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)

# use this cross validation method in GridSearchCV:
grid_search = GridSearchCV(pipe, param_grid = parameters, cv=sk_fold, scoring = 'f1')

# with 'StratifiedShuffleSplit' you fit the complete dataset
# GridSearchCV, internally, will use the indices from 'StratifiedShuffleSplit'
# to fit all 100 folds (all 100 test/train subsets).
grid_search.fit(features, labels)

# extract the best algorithm:
clf = grid_search.best_estimator_

print 'best algorithm using strat_s_split: '
print clf

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
score = clf.score(features_test, labels_test)
print score

pre = precision_score(labels_test, pred)
print "precision: ",pre
rec = recall_score(labels_test, pred)
print "recall: ",rec


dump_classifier_and_data(clf, my_dataset, features_list)
