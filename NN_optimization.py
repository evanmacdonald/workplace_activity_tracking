# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:00:08 2018

@author: Evan Macdonald

Trains SVM classification algorithm using train data and then outputs results from test data
"""

#Imports
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
import Kintec_Functions as kf
import neural_network as nn


#%% import all data
# NOTE: SID21,22,23 do not have calibration data
# NOTE: SID18 and SID27 should be left out as it's data causes issues for the algorithm
print('Importing Data...')

comp = 'evanm'

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID01/SynchronizedData/SID01_Calibration.csv')
SID01 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID02/SynchronizedData/SID02_Calibration.csv')
SID02 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID03/SynchronizedData/SID03_Calibration.csv')
SID03 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID04/SynchronizedData/SID04_Calibration.csv')
SID04 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID05/SynchronizedData/SID05_Calibration.csv')
SID05 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID06/SynchronizedData/SID06_Calibration.csv')
SID06 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID07/SynchronizedData/SID07_Calibration.csv')
SID07 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID08/SynchronizedData/SID08_Calibration.csv')
SID08 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID09/SynchronizedData/SID09_Calibration.csv')
SID09 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID10/SynchronizedData/SID10_Calibration.csv')
SID10 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID11/SynchronizedData/SID11_Calibration.csv')
SID11 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID12/SynchronizedData/SID12_Calibration.csv')
SID12 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID13/SynchronizedData/SID13_Calibration.csv')
SID13 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID14/SynchronizedData/SID14_Calibration.csv')
SID14 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID15/SynchronizedData/SID15_Calibration.csv')
SID15 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID16/SynchronizedData/SID16_Calibration.csv')
SID16 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID17/SynchronizedData/SID17_Calibration.csv')
SID17 = pd.read_csv(file)

#leave this one out, it is not good data
file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID18/SynchronizedData/SID18_Calibration.csv')
SID18 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID19/SynchronizedData/SID19_Calibration.csv')
SID19 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID20/SynchronizedData/SID20_Calibration.csv')
SID20 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID24/SynchronizedData/SID24_Calibration.csv')
SID24 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID25/SynchronizedData/SID25_Calibration.csv')
SID25 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID26/SynchronizedData/SID26_Calibration.csv')
SID26 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID28/SynchronizedData/SID28_Calibration.csv')
SID28 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID29/SynchronizedData/SID29_Calibration.csv')
SID29 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID30/SynchronizedData/SID30_Calibration.csv')
SID30 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID31/SynchronizedData/SID31_Calibration.csv')
SID31 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID32/SynchronizedData/SID32_Calibration.csv')
SID32 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID33/SynchronizedData/SID33_Calibration.csv')
SID33 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID34/SynchronizedData/SID34_Calibration.csv')
SID34 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID35/SynchronizedData/SID35_Calibration.csv')
SID35 = pd.read_csv(file)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID36/SynchronizedData/SID36_Calibration.csv')
SID36 = pd.read_csv(file)

print('Data imported...')

#%% Leave one out setup

#modify this list to determine which Subjects to include in the training / testing data.
# This vvv is a complete list, copy this then modify as needed.
#SID_i = ['SID01','SID02','SID03','SID04','SID05','SID06','SID07','SID08','SID09','SID10',
#         'SID11','SID12','SID13','SID14','SID15','SID16','SID17','SID18','SID19','SID20',
#         'SID24','SID25','SID26','SID28','SID29','SID30','SID31','SID32','SID33','SID34','SID35','SID36']

SID_i = ['SID05','SID06','SID07','SID08','SID09','SID10',
         'SID12','SID13','SID14','SID15','SID16','SID17','SID19','SID20',
         'SID24','SID25','SID26','SID30','SID32','SID34','SID36']

#SID_i = ['SID05','SID06']

#Dictionary pointing to all the dataframes imported above to be used for iterating through.
SID_dict = {'SID01':SID01,'SID02':SID02,'SID03':SID03,'SID04':SID04,'SID05':SID05,'SID06':SID06,'SID07':SID07,'SID08':SID08,'SID09':SID09,'SID10':SID10,
            'SID11':SID11,'SID12':SID12,'SID13':SID13,'SID14':SID14,'SID15':SID15,'SID16':SID16,'SID17':SID17,'SID18':SID18,'SID19':SID19,'SID20':SID20,
            'SID24':SID24,'SID25':SID25,'SID26':SID26,'SID28':SID28,'SID29':SID29,'SID30':SID30,'SID31':SID31,'SID32':SID32,'SID33':SID33,'SID34':SID34,'SID35':SID35,'SID36':SID36}

#Format for calling a specific subject's data
#test_data = SID_dict[SID_i[i]]

#%% test neural network with one training set
#
#test_data = SID_dict[SID_i[0]]
#train_data = SID_dict[SID_i[1]]
#
#print('Buffering Data...')
## buffer data for input into classifier
#buffLen =  30 #45, 90, 135, 180, 225
#numVars = 20
##select_feats = np.arange(106) #all features
#select_feats = np.asarray([74,76,22,27,65,56,16,73,33,14,5,71,88,97,26,70,62,63,64,78])
#
##train data [*,numFeats]
#train_data = kf.normalize_3(train_data)
#train_segments, train_labels = kf.segment_values(train_data,buffLen)
#train_x = train_segments.reshape(len(train_segments), 1, buffLen, numVars)
#train_x = kf.get_max_features(train_x)
#train_x = np.take(train_x,select_feats,axis=1)
#train_y = np.asarray(pd.get_dummies(train_labels), dtype = np.int8)
#
##test data
#test_data = kf.normalize_3(test_data)
#test_segments, test_labels = kf.segment_values(test_data,buffLen)
#test_x = test_segments.reshape(len(test_segments), 1, buffLen, numVars)
#test_x = kf.get_max_features(test_x)
#test_x = np.take(test_x,select_feats,axis=1)
#test_y = np.asarray(pd.get_dummies(test_labels), dtype = np.int8)
#test_y_cls = np.argmax(test_y, axis=1)
#
#print('Data buffered...')

#%%
#num_nodes = 500
#acc_train, acc_test, cost_step, total_iterations = nn.train_neural_net(train_x,train_y,test_x,test_y,test_y_cls,select_feats,num_nodes,buffLen)

##%% Get predictions for full feature set based on a Leave One Out cross validation
#print('Getting solutions...')
#
## Declared Variables
#buffLen = 30 #45, 90, 135, 180, 225
#numVars = 20
#num_nodes = 40
##select_feats = np.arange(106) #all features
#select_feats = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33]) #FSRs Mean only
#
##Lists for saving data
#test_data_id = []
#train_data_lst = []
#prediction_lst = []
#solution_lst = []
#acc_train_lst = []
#acc_test_lst = []
#
## Get results for each participant's data using leave one out cross validation.
## Each iteration selects a new dataset for testing and creates a training set with the rest fo the datasets 
#loo = LeaveOneOut()
#for train_index, test_index in loo.split(SID_i):
#    # Reference lists
#    test_data_id.append(SID_i[test_index[0]])
#    train_data_lst.append(train_index)
#    
#    #get raw training and testing data
#    test_data = SID_dict[SID_i[test_index[0]]]
#    train_data = SID_dict[SID_i[train_index[0]]]
#    for i in train_index[1:]:
#        df = SID_dict[SID_i[i]]
#        train_data = train_data.append(df).reset_index(drop=True)
#    
#    #pre-process train and test data appropriately
#    #train data [*,numFeats]
#    train_data = kf.normalize_3(train_data)
#    train_segments, train_labels = kf.segment_values(train_data,buffLen)
#    train_x = train_segments.reshape(len(train_segments), 1, buffLen, numVars)
#    train_x = kf.get_max_features(train_x)
#    train_x = np.take(train_x,select_feats,axis=1)
#    train_y = np.asarray(pd.get_dummies(train_labels), dtype = np.int8)
#    
#    #test data
#    test_data = kf.normalize_3(test_data)
#    test_segments, test_labels = kf.segment_values(test_data,buffLen)
#    test_x = test_segments.reshape(len(test_segments), 1, buffLen, numVars)
#    test_x = kf.get_max_features(test_x)
#    test_x = np.take(test_x,select_feats,axis=1)
#    test_y = np.asarray(pd.get_dummies(test_labels), dtype = np.int8)
#    test_y_cls = np.argmax(test_y, axis=1)
#    
#    #get results
#    acc_train, acc_test, cost_step, total_iterations, y_ = nn.train_neural_net(train_x,train_y,test_x,test_y,test_y_cls,select_feats,num_nodes,buffLen)
#    prediction_lst.append(y_)
#    solution_lst.append(test_labels)  
#    acc_train_lst.append(acc_train)
#    acc_test_lst.append(acc_test)
#    print('   ',SID_i[test_index[0]],' Completed')
#        
#print('Solutions obtained...')
#
## Put accuracy next to the ID name in a dataframe for further analysis
#results = pd.DataFrame({'Test Data ID':test_data_id, 'Accuracy':acc_test_lst})

#%% Optimize NN using Leave One Out Cross Validation

print('Getting solutions...')
#parameters to optimize
buffLen = [20,30,45,90,135,180] #10,20,30,45,90,135,180
num_nodes = [20,30,40,50,60,70,80,90,100,150,200] #20,30,40,50,60,70,80,90,100,150,200

combos = np.array(np.meshgrid(buffLen,num_nodes)).T.reshape(-1,2)

numVars = 20
#select_feats = np.arange(106) #all features
select_feats = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,30,31,32,33]) #FSRs Mean only

#initialize blank solution arrays to store results
trainAccList = []
testAccList = []
iterList = []
costList = []

for combo in range(len(combos)):
    
    #Lists for saving data
    test_data_id = []
    acc_train_lst = []
    acc_test_lst = []
    iter_lst = []
    cost_lst = []
    
    buffLen = combos[combo,0]
    num_nodes = combos[combo,1]
    
    # Get results for each participant's data using leave one out cross validation.
    # Each iteration selects a new dataset for testing and creates a training set with the rest fo the datasets 
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(SID_i):
        # Reference lists
        test_data_id.append(SID_i[test_index[0]])
        
        #get raw training and testing data
        test_data = SID_dict[SID_i[test_index[0]]]
        train_data = SID_dict[SID_i[train_index[0]]]
        for i in train_index[1:]:
            df = SID_dict[SID_i[i]]
            train_data = train_data.append(df).reset_index(drop=True)
        
        #pre-process train and test data appropriately
        #train data [*,numFeats]
        train_data = kf.normalize_3(train_data)
        train_segments, train_labels = kf.segment_values(train_data,buffLen)
        train_x = train_segments.reshape(len(train_segments), 1, buffLen, numVars)
        train_x = kf.get_max_features(train_x)
        train_x = np.take(train_x,select_feats,axis=1)
        train_y = np.asarray(pd.get_dummies(train_labels), dtype = np.int8)
        
        #test data
        test_data = kf.normalize_3(test_data)
        test_segments, test_labels = kf.segment_values(test_data,buffLen)
        test_x = test_segments.reshape(len(test_segments), 1, buffLen, numVars)
        test_x = kf.get_max_features(test_x)
        test_x = np.take(test_x,select_feats,axis=1)
        test_y = np.asarray(pd.get_dummies(test_labels), dtype = np.int8)
        test_y_cls = np.argmax(test_y, axis=1)
        
        #get results
        acc_train, acc_test, cost_step, total_iterations, y_ = nn.train_neural_net(train_x,train_y,test_x,test_y,test_y_cls,select_feats,num_nodes,buffLen)
        acc_train_lst.append(acc_train)
        acc_test_lst.append(acc_test)
        iter_lst.append(total_iterations)
        cost_lst.append(cost_step)
#        print('   ',SID_i[test_index[0]],' Completed')
    
    trainAccList.append(acc_train_lst)
    testAccList.append(acc_test_lst)
    iterList.append(iter_lst)
    costList.append(cost_lst)
    
    msg = "Completed combination #: {0:>2} of {1:>2}"
    print(msg.format((combo+1), len(combos)))
