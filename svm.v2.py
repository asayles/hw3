

"""
linear kearnal

split training into 10 equal sets

for C_parameter in {0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1}
    for set in  traiging_sets
        set = validation_set
        train svm (using C_parameter) with each non-validation set
        Test model on validation_set
    average accuracy
Choose the C_parameter with the highest accuracy

train new linear svm using all training dataand chosen C_parameter

Test model on test data
    report accuracy, precision, and recall
        using threshold 0 to determine pos / neg classification
        
Use results to create ROC using 200 evenly spaced thresholds
        
""""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

for C_parameter in {0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1}:
    for X in training_sets:
        # X : {array-like, sparse matrix}, shape (n_samples, n_features)
        # y : array-like, shape (n_samples,)
        # sample_weight : array-like, shape (n_samples,)
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
        clf = svm.SVC(kernel='linear', C=C_parameter)
        clf.fit(X,y)
        
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linespace(-5, 5)