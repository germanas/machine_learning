#!/usr/bin/python

import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def find_empty(dataset):
    NAN = 0
    TOTAL = 0
    for person in dataset:
        for key in dataset[person].keys():
            if dataset[person][key] == 'NaN':
                NAN += 1
                TOTAL += 1
            else:
                TOTAL += 1
    print 'Total feature values missing: ', NAN
    print 'Total feature values: ', TOTAL
    print 'The percentage compared to all values: ', float(NAN)/TOTAL * 100


def plotting_salary_expenses(dataset, feature1, feature2):
    '''this function takes dataset, 2 features and creates a scatter plot from these features'''
    x = []
    y = []
    z = []
    new_z = []
    for person in dataset:
        for feature in dataset[person]:
            if feature == feature1:
                x.append(dataset[person][feature])
            elif feature == feature2:
                y.append(dataset[person][feature])
            elif feature == 'poi':
                z.append(dataset[person][feature])
    for i in z:
        if i == True:
            new_z.append('g')
        elif i == False:
            new_z.append('r')
    fig, ax = plt.subplots()
    for color in new_z:
        ax.scatter(x, y, c=new_z)


    red_patch = mpatches.Patch(color='red', label='Not poi')
    green_patch = mpatches.Patch(color='green', label='Poi')
    ax.legend(handles=[red_patch, green_patch])
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    plt.show()

def dataset_info(dataset):
    '''Takes in a dataset and prints out the information of this dataset'''
    #Print number of data points on dataset
    print 'Number of total datapoints: ', len(dataset)
    #Print number of features for each datapoint
    available_features = set()
    total = 0
    for key in dataset.iterkeys():
        for item in dataset[key]:
            available_features.add(item)

    print 'Number of features for each datapoint: ', len(available_features)
    #print 'Those features are: '
    #pprint.pprint(available_features)
    # Find the number of POI's in the dataset
    poi_count = 0
    not_poi_count = 0
    for key in dataset:
        if dataset[key]['poi'] == True:
            poi_count += 1
        elif dataset[key]['poi'] == False:
            not_poi_count += 1
    print 'Number of persons of interest in this dataset: ', poi_count
    print 'Number of other people in this dataset: ', not_poi_count

def remove_outlier(dataset, keys):
    '''Takes in a dataset and a list of outlier keys, and removes them from dataset'''
    for key in keys:
        dataset.pop(key, 0)

def ratio(poi_to, poi_from):
    ''' Calculates the ratio between poi_to and poi_from emails.'''
    if poi_to == 'NaN' or poi_from == 'NaN':
        return 0
    elif poi_to == 0 or poi_from == 0:
        return 0
    ratio = float(poi_to) / poi_from
    return ratio

