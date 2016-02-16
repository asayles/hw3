
from data_processing import magic_data_scaler as mds
from svm_expr1_v2 import find_best_C_param
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import heapq, numpy
"""
use model from expr1
obtain weight vector w
select features
    
    
create list of indecies of largest weight
create copy of selected features
train model
"""
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
feature_selection = []
for m in range(2,57):
    index_of_selected_features = heapq.nlargest(m,range(len(weight_vals[0])),weight_vals[0].take)
    X_train_feat_scaled_selected = numpy.copy(X_train_feat_scaled[:,[eye for eye in index_of_selected_features]])
    
    super_model = svm.SVC(kernel='linear', C=best_C_param)
    super_model.fit(X_train_feat_scaled_selected,Y_train_class)
    super_model_accuracy.append(super_model.score(X_test_feat_scaled[:,[eye for eye in index_of_selected_features]], Y_test_class))
    super_model_predict = super_model.predict(X_test_feat_scaled[:,[eye for eye in index_of_selected_features]])
    
    feature_selection.append(index_of_selected_features)
    
  
# plt.plot(range(2, 57), super_model_accuracy, label='Accuracy')
# plt.xlim([0.0, 59.0])
# plt.ylim([0.6, 1.05])
# plt.xlabel('Num features')
# plt.ylabel('Accuracy of model')
# plt.title('Expr 2: accuracy vs. m features')
# plt.legend(loc="lower right")
# plt.show()

# features from spambase.names
    feature_vals = ["word_freq_make",
                    "word_freq_address",
                    "word_freq_all",
                    "word_freq_3d",
                    "word_freq_our",
                    "word_freq_over",
                    "word_freq_remove",
                    "word_freq_internet",
                    "word_freq_order",
                    "word_freq_mail",
                    "word_freq_receive",
                    "word_freq_will",
                    "word_freq_people",
                    "word_freq_report",
                    "word_freq_addresses",
                    "word_freq_free",
                    "word_freq_business",
                    "word_freq_email",
                    "word_freq_you",
                    "word_freq_credit",
                    "word_freq_your",
                    "word_freq_font",
                    "word_freq_000",
                    "word_freq_money",
                    "word_freq_hp",
                    "word_freq_hpl",
                    "word_freq_george",
                    "word_freq_650",
                    "word_freq_lab",
                    "word_freq_labs",
                    "word_freq_telnet",
                    "word_freq_857",
                    "word_freq_data",
                    "word_freq_415",
                    "word_freq_85",
                    "word_freq_technology",
                    "word_freq_1999",
                    "word_freq_parts",
                    "word_freq_pm",
                    "word_freq_direct",
                    "word_freq_cs",
                    "word_freq_meeting",
                    "word_freq_original",
                    "word_freq_project",
                    "word_freq_re",
                    "word_freq_edu",
                    "word_freq_table",
                    "word_freq_conference",
                    "char_freq_;",
                    "char_freq_(",
                    "char_freq_[",
                    "char_freq_!",
                    "char_freq_$",
                    "char_freq_# ",
                    "capital_run_length_average",
                    "capital_run_length_longest",
                    "capital_run_length_total"]

    max_index = super_model_accuracy.index(max(super_model_accuracy))

    # get features selected when accuracy of SVM was highest
    # print "Features selected with highest accuracy:",features_selected[max_index]
    # store features selected to display feature values
    print "Feature values for features with highest accuracy:"
    for i in feature_selection[max_index]:
        print feature_vals[i]