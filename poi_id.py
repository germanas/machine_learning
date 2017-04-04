#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

### Helper functions

#plot features ####!!!! rebuild this function to take any 2 features and display a scatterplot
def plotting_salary_expenses(dataset, feature1, feature2):
    #print dataset
    x = []
    y = []
    z = []
    new_z = []
    for person in dataset:
        for feature in data_dict[person]:
            if feature == feature1:
                x.append(data_dict[person][feature])
            elif feature == feature2:
                y.append(data_dict[person][feature])
            elif feature == 'poi':
                z.append(data_dict[person][feature])
    for i in z:
        if i == True:
            new_z.append('g')
        elif i == False:
            new_z.append('r')
    fig, ax = plt.subplots()
    for color in new_z:
        ax.scatter(x, y, c=new_z)
    #plt.scatter(x,y, c=z, label=z)
    red_patch = mpatches.Patch(color='red', label='Poi TRUE')
    green_patch = mpatches.Patch(color='green', label='Poi FALSE')
    ax.legend(handles=[red_patch, green_patch])
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','expenses'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#my_dataset = data_dict
print data_dict["TOTAL"]
### Task 2: Remove outliers
plotting_salary_expenses(data_dict, 'salary', 'expenses')
data_dict.pop("TOTAL", 0)
plotting_salary_expenses(data_dict, 'salary', 'expenses')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
