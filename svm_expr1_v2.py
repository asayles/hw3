

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
from sklearn import svm, datasets
from sklearn import svm, cross_validation, metrics
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc
from data_processing import magic_data_scaler as mds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def find_best_C_param():
    X_train_feat_split = numpy.split(X_train_feat_scaled,10,axis=0)
    Y_train_class_split = numpy.split(Y_train_class,10,axis=0)

    accuracy_avgs = []
    C_parameter_superset = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    for C_parameter in C_parameter_superset:
        # print "new C_parameter: ", C_parameter
        accuracy_raw = []
        
        # create cross-validation subsets from training superset
        for index_of_val_set in range(len(X_train_feat_split)):
            validation_features = X_train_feat_split[index_of_val_set]
            validation_class = Y_train_class_split[index_of_val_set]
            
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

            train_class = numpy.concatenate((Y_train_class_split[index_subset[0]],
                                            Y_train_class_split[index_subset[1]],
                                            Y_train_class_split[index_subset[2]],
                                            Y_train_class_split[index_subset[3]],
                                            Y_train_class_split[index_subset[4]],
                                            Y_train_class_split[index_subset[5]],
                                            Y_train_class_split[index_subset[6]],
                                            Y_train_class_split[index_subset[7]],
                                            Y_train_class_split[index_subset[8]]))
        
            clf = svm.SVC(kernel='linear', C=C_parameter)
            clf.fit(train_features,train_class)
            accuracy = clf.score(validation_features, validation_class)
            accuracy_raw.append(accuracy)
        # average all the accuracy and store
        accuracy_avgs.append(sum(accuracy_raw) / len(accuracy_raw))
        # print "\rC_param: " + str(C_parameter) + " accuracy: " + str(accuracy_raw)
    # print accuracy_avgs
    return C_parameter_superset[accuracy_avgs.index(max(accuracy_avgs))]

def print_metrics():
    print metrics.classification_report(Y_test_class, model_predict, target_names=['NotSoSpammy','Spamilicious'])
    
#-----------------
# get the data's
X_train_feat_scaled,Y_train_class,X_test_feat_scaled,Y_test_class = mds()

#-------------------------------------
# find C_param with highest accuracy
best_C_param = find_best_C_param()

#--------------------------------------------------------      
# train the model with all tes data and the best_C_param    
model = svm.SVC(kernel='linear', C=best_C_param)
model.fit(X_train_feat_scaled,Y_train_class)
model_score = model.fit(X_train_feat_scaled,Y_train_class).decision_function(X_test_feat_scaled)
model_accuracy = model.score(X_test_feat_scaled,Y_test_class)
model_predict = model.predict(X_test_feat_scaled)


if __name__ == "__main__":
    #-------------------      
    # print some stuff
    print "the best C for this run is: ", best_C_param
    print_metrics()  
      
    #---------------
    # ROC hotness

    # Import some data to play with
    all_the_data = numpy.concatenate((X_train_feat_scaled, X_test_feat_scaled),axis=0)
    all_the_class = numpy.concatenate((Y_train_class[:, None], Y_test_class[:, None]),axis=0)

    # Binarize the output
    all_the_class_binarized = label_binarize(all_the_class, classes=[0,1])
    n_classes = all_the_class_binarized.shape[1]

    # shuffle and split training and test sets
    X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(all_the_data, all_the_class_binarized, test_size=.5, random_state=0)

    # Learn to predict each class against the other
    classy_flier = OneVsRestClassifier(svm.SVC(kernel='linear', C=best_C_param, probability=True))
    y_score_roc = classy_flier.fit(X_train_roc, y_train_roc).decision_function(X_test_roc)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score_roc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score_roc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
                                            ''.format(roc_auc["micro"]), linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'
                                            ''.format(roc_auc["macro"]), linewidth=2)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Exper 1 ROC')
    plt.legend(loc="lower right")
    plt.show()







