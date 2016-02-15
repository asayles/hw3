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
    
def ConvertDataToArrays(features, labels):
  """
  conversion to a numpy array is easy if you're starting with a List of lists.
  The returned array has dimensions (M,N), where M is the number of lists and N is the number of 
  
  """

  return np.asarray(features), np.asarray(labels)
  

def NormalizeFeatures(features):
  """
  I'm providing this mostly as a way to demonstrate array operations using Numpy.  Incidentally it also solves a small step in the homework.
  """
  
  "selecting axis=0 causes the mean to be computed across each feature, for all the samples"
  means = np.mean(features, axis = 0)
  
  variances = np.var(features, axis = 0)

  "Operations in numpy performed on a 2D array and a 1D matrix will automatically broadcast correctly, if the leading dimensions match."
  features = features - means
  # features -= means

  features /= variances

  return features


def PrintDataToSvmLightFormat(features, labels, filename = "svm_features.data"):
  """  
  Readable format for SVM Light should be, with 
  lable 0:feature0, 1:feature1, 2:feature2, etc...
  where label is -1 or 1.
  """

  if len(features) != len(labels):
    raise Exception("Number of samples and labels must match")
  dat_file = file(filename,'w')
  for s in range(len(features)):

    if labels[s]==0:
      line="-1 "
    else:
      line="1 "

    for f in range(len(features[s])):
      line +="%i:%f " % (f+1 , features[s][f])
    line += "\n"
    dat_file.write(line)
  dat_file.close()


def FeatureSubset(features_array, indices):
  """
  Takes the original set of features and returns a small array containing only the features with the given indices.

  features_array is a numpy 2D array of dimension (M,N), where M is the number of samples and N is the number of features in the feature vector.

  indices are those of the features to be used, as a list of integers  
  """
  return features_array[:,indices]

def main():

  features, labels = LoadSpamData()
  features, labels = BalanceDataset(features, labels)
  features, labels = ConvertDataToArrays(features, labels)
  features = NormalizeFeatures(features)
  indices = [0,1,2]
  features = FeatureSubset(features, indices)

  PrintDataToSvmLightFormat(features, labels)

if __name__ == "__main__":
  main()