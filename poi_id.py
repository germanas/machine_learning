#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Import helper functions
from helper_functions import plotting_salary_expenses, dataset_info, remove_outlier, ratio

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
#plotting_salary_expenses(data_dict, 'salary', 'expenses')
#plotting_salary_expenses(data_dict, 'salary', 'from_poi_to_this_person')
#plotting_salary_expenses(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
#plotting_salary_expenses(data_dict, 'salary', 'from_this_person_to_poi')
#plotting_salary_expenses(data_dict, 'shared_receipt_with_poi', 'to_messages')
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

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# I will now use k-best to find 5 most promising features
from sklearn.feature_selection import SelectKBest
features = SelectKBest(k=10).fit_transform(features, labels)

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

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#Creating a function to test and tune my algorithm
clf = clf2
#clf1.fit(lab)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
