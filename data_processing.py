#!/usr/bin/python

import pandas
import numpy as np
from sklearn import preprocessing as pp

"""
Read file 
Split into pos,neg classified sets
Truncate the larger set
Join half the pos and neg data as training, test
Shuffle both data sets to eliminate ordering
Scale the data
    Create float matrix
    Create a scaled copy of matrix
    Create array of classifications
    Repeat for test
Return train_feat_scaled, train_class, test_feat_scaled, test_class
"""
def magic_data_scaler():
    data_raw = pandas.read_csv('/home/ajsayles/School/cs445/data_raw.csv', header=None)

    class_pos = (data_raw.loc[data_raw[57] == 1])
    # print class_pos.shape[0]
    pos = class_pos.shape[0]
    class_neg = (data_raw.loc[data_raw[57] == 0])
    # print class_neg.shape[0]
    neg = class_neg.shape[0]

    if pos > neg:
        # impliment later if there is time
        print "more positive classifications" 
    else:
        # print "more neg examples"
        class_neg = class_neg[0:(pos -1)].reset_index(drop=True)
        

    training_superset = [class_pos[0:(pos / 2) - 1], class_neg[0:(pos / 2) - 1]]
    training_frame = pandas.concat(training_superset)
    training_frame = training_frame.sample(frac=1).reset_index(drop=True)
    # print training_frame

    test_superset = [class_pos[(pos /2):pos - 2], class_neg[(pos / 2):pos - 2]]
    test_frame = pandas.concat(test_superset)
    test_frame = test_frame.sample(frac=1).reset_index(drop=True)
    # print test_frame

    training_frame = training_frame.sample(frac=1).reset_index(drop=True)
    test_frame = test_frame.sample(frac=1).reset_index(drop=True)
    # print training_frame

    X_train = training_frame.as_matrix().astype(np.float)
    # print X
    X_train_feat_scaled = pp.StandardScaler().fit_transform(X_train[:,:57])
    # print X_train_feat_scaled
    X_train_class = X_train[:,57]
    # print X_train_class

    X_test = test_frame.as_matrix().astype(np.float)
    # print X_test
    X_test_feat_scaled = pp.StandardScaler().fit_transform(X_test[:,:57])
    # print X_test_feat_scaled
    X_test_class = X_test[:,57]
    # print X_test_class
    
    return X_train_feat_scaled,X_train_class,X_test_feat_scaled,X_test_class
