#!/usr/bin/python

import csv
import numpy as np


def load_spam_data(filename = 'data_raw'):
    
    file_data_raw = file(filename,'rb')
    data_raw = file_data_raw.readlines()
    
    labels = []
    features = []
    
    for line in data_raw:
        feature_vector = []
        split_line = line.split(',')
        
        for element in split_line[:-1]:
            feature_vector.append(float(element))
            
        features.append(feature_vector)
        labels.append(int(split_line[-1]))
        
    return features, labels
    
def balance_dataset(features, labels):

    count_0 = labels.count(0)
    count_1 = labels.count(1)
    balanced_count = min(count_0, count_1)
    
    return features[:balanced_count] + features[-balanced_count], labels[:balanced_count] + labels[-balanced_count:]
    
    
