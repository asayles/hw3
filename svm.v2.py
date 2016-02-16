

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
        
"""
import numpy, sys
import matplotlib.pyplot as plt
from sklearn import svm
from data_processing import magic_data_scaler as mds

# get the data's
#-----------------
X_train_feat_scaled,X_train_class,X_test_feat_scaled,X_test_class = mds()

# find C_param with highest accuracy
#-------------------------------------
X_train_feat_split = numpy.split(X_train_feat_scaled,10,axis=0)
X_train_class_split = numpy.split(X_train_class,10,axis=0)

accuracy_avgs = []
for C_parameter in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
    # print "new C_parameter: ", C_parameter
    accuracy_raw = []
    
    # create cross-validation subsets from training superset
    for index_of_val_set in range(len(X_train_feat_split)):
        validation_features = X_train_feat_split[index_of_val_set]
        validation_class = X_train_class_split[index_of_val_set]
         
        index_subset = range(10)
        index_subset.pop(index_of_val_set)        
        # print index_of_val_set, index_subset
        
        train_features = numpy.concatenate((X_train_feat_split[index_subset[0]],
                                            X_train_feat_split[index_subset[1]],
                                            X_train_feat_split[index_subset[2]],
                                            X_train_feat_split[index_subset[3]],
                                            X_train_feat_split[index_subset[4]],
                                            X_train_feat_split[index_subset[5]],
                                            X_train_feat_split[index_subset[6]],
                                            X_train_feat_split[index_subset[7]],
                                            X_train_feat_split[index_subset[8]]))

        train_class = numpy.concatenate((X_train_class_split[index_subset[0]],
                                        X_train_class_split[index_subset[1]],
                                        X_train_class_split[index_subset[2]],
                                        X_train_class_split[index_subset[3]],
                                        X_train_class_split[index_subset[4]],
                                        X_train_class_split[index_subset[5]],
                                        X_train_class_split[index_subset[6]],
                                        X_train_class_split[index_subset[7]],
                                        X_train_class_split[index_subset[8]]))
      
        clf = svm.SVC(kernel='linear', C=C_parameter)
        clf.fit(train_features,train_class)
        accuracy = clf.score(validation_features, validation_class)
        accuracy_raw.append(accuracy)
    # average all the accuracy and store
    accuracy_avgs.append(sum(accuracy_raw) / len(accuracy_raw))
    # print "\rC_param: " + str(C_parameter) + " accuracy: " + str(accuracy_raw)
print accuracy_avgs
best_C_param = accuracy_avgs.index(max(accuracy_avgs))
print best_C_param
        # w = clf.coef_[0]
        # a = -w[0] / w[1]
        # xx = np.linespace(-5, 5)
        # yy = a * xx - clf.intercept_[0] / w[1]
        
        
#         print w
# # # get the separating hyperplane using weighted classes
# # wclf = svm.SVC(kernel='linear', class_weight={1: 10})
# # wclf.fit(X, y)

# # ww = wclf.coef_[0]
# # wa = -ww[0] / ww[1]
# # wyy = wa * xx - wclf.intercept_[0] / ww[1]

# # # plot separating hyperplanes and samples
# # h0 = plt.plot(xx, yy, 'k-', label='no weights')
# # h1 = plt.plot(xx, wyy, 'k--', label='with weights')
# # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
# # plt.legend()

# # plt.axis('tight')
# # plt.show()