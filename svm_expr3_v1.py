from data_processing import magic_data_scaler as mds
from svm_expr1_v2 import find_best_C_param
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import heapq, numpy

#-----------------
# get the data's
X_train_feat_scaled,Y_train_class,X_test_feat_scaled,Y_test_class = mds()

#-------------------------------------
# find C_param with highest accuracy
best_C_param = find_best_C_param()

#--------------------------------------------------------      
# train the model with all test data and the best_C_param
OG_model = svm.SVC(kernel='linear', C=best_C_param)
OG_model.fit(X_train_feat_scaled,Y_train_class)
OG_model_score = OG_model.fit(X_train_feat_scaled,Y_train_class).decision_function(X_test_feat_scaled)
OG_model_accuracy = OG_model.score(X_test_feat_scaled,Y_test_class)
OG_model_predict = OG_model.predict(X_test_feat_scaled)

#-------------------------------
# get weight coef's
weight_vals = numpy.absolute(OG_model._get_coef())
# print weight_vals

super_model_accuracy = []

for m in range(2,57):
    index_of_rand_selected_features = numpy.random.choice(range(len(weight_vals[0])), m, replace=False)
    X_train_feat_scaled_selected = numpy.copy(X_train_feat_scaled[:,[eye for eye in index_of_rand_selected_features]])
    
    super_model = svm.SVC(kernel='linear', C=best_C_param)
    super_model.fit(X_train_feat_scaled_selected,Y_train_class)
    super_model_accuracy.append(super_model.score(X_test_feat_scaled[:,[eye for eye in index_of_rand_selected_features]], Y_test_class))
    super_model_predict = super_model.predict(X_test_feat_scaled[:,[eye for eye in index_of_rand_selected_features]])

#---------------------
# plot the things


plt.plot(range(2, 57), super_model_accuracy, label='acc')
plt.xlim([1.0, 59.0])
plt.ylim([0.0, 1.05])
# label x and y axes, generate title
plt.xlabel('num features')
plt.ylabel('accuracy of SVM')
plt.title('accuracy vs. "m" features (random selection)')
plt.legend(loc="lower right")
plt.show()