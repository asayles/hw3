

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

# we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)


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
        yy = a * xx - clf.intercept_[0] / w[1]
        
        
        print w
# # get the separating hyperplane using weighted classes
# wclf = svm.SVC(kernel='linear', class_weight={1: 10})
# wclf.fit(X, y)

# ww = wclf.coef_[0]
# wa = -ww[0] / ww[1]
# wyy = wa * xx - wclf.intercept_[0] / ww[1]

# # plot separating hyperplanes and samples
# h0 = plt.plot(xx, yy, 'k-', label='no weights')
# h1 = plt.plot(xx, wyy, 'k--', label='with weights')
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
# plt.legend()

# plt.axis('tight')
# plt.show()