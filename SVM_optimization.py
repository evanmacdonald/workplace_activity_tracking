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
#
SID_i = ['SID05','SID06','SID07','SID08','SID09','SID10',
         'SID12','SID13','SID14','SID15','SID16','SID17','SID19','SID20',
         'SID24','SID25','SID26','SID30','SID32','SID34','SID36']

#Dictionary pointing to all the dataframes imported above to be used for iterating through.
SID_dict = {'SID01':SID01,'SID02':SID02,'SID03':SID03,'SID04':SID04,'SID05':SID05,'SID06':SID06,'SID07':SID07,'SID08':SID08,'SID09':SID09,'SID10':SID10,
            'SID11':SID11,'SID12':SID12,'SID13':SID13,'SID14':SID14,'SID15':SID15,'SID16':SID16,'SID17':SID17,'SID18':SID18,'SID19':SID19,'SID20':SID20,
            'SID24':SID24,'SID25':SID25,'SID26':SID26,'SID28':SID28,'SID29':SID29,'SID30':SID30,'SID31':SID31,'SID32':SID32,'SID33':SID33,'SID34':SID34,'SID35':SID35,'SID36':SID36}

#Format for calling a specific subject's data
#test_data = SID_dict[SID_i[i]]


#%% Optimize SVM parameters using Leave One Out Cross Validation

print('Getting solutions...')

#parameters to optimize. Make array of potential values for each value
buffLen = [40] #10,20,30,45,90,135,180
C = [105] #1,2,3,4,5,6,7,8,9,10
#make combination list of all possible combinations of parameters
combos = np.array(np.meshgrid(buffLen,C)).T.reshape(-1,2)


numVars = 16 #should always be 20, the number of sensors
#select what features to use. See 'Features Legend.xlsx' for details
select_feats = np.asarray([0,7,16,23, #FSR1
                           1,8,17,24, #FSR2
                           2,9,18,25, #FSR3
                           3,10,19,26, #FSR4
                           4,11,20,27, #FSR5
                           5,12,21,28, #FSR6
                           6,13,22,29, #FSR7
                           14,15,30,31,46,47,62,63,78,79]) #FSRs only, Mean and STD

testAccList = []

for combo in range(len(combos)):
    
    #Lists for saving data
    solution_lst = []
    test_data_id = []
    accuracy_lst = []
    
    buffLen = combos[combo,0]
    C = combos[combo,1]
    
    # Get results for each participant's data using leave one out cross validation.
    # Each iteration selects a new dataset for testing and creates a training set with the rest fo the datasets 
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(SID_i):
        # Reference lists
        test_data_id.append(SID_i[test_index[0]])
        
        #get training and testing data
        test_data = SID_dict[SID_i[test_index[0]]]
        train_data = SID_dict[SID_i[train_index[0]]]
        for j in train_index[1:]:
            df = SID_dict[SID_i[j]]
            train_data = train_data.append(df).reset_index(drop=True)
        
        #train model
        X_train, y_train = kf.pre_process_data(train_data,order=2,cutoff=10,buffLen=buffLen,numvars=numVars)
        X_train = np.take(X_train,select_feats,axis=1)
        model = svm.SVC(kernel='poly', gamma=0.05, C=C)
        model.fit(X_train,y_train)
        
        #test model
        X_test, y_test = kf.pre_process_data(test_data,order=2,cutoff=10,buffLen=buffLen,numvars=numVars)
        X_test = np.take(X_test,select_feats,axis=1)
        y_ = model.predict(X_test)
        solution_lst.append(y_test)  
        
        #get accuracy
        y_out_val = kf.reshape_ybar(y_,test_data)
        test_labels = np.asarray(test_data['ActivityState'],dtype='int32')
        title = 'blank'
        a, b = kf.get_cm(test_labels, y_out_val, title)
        accuracy_lst.append(b)
        
#        print('   ',SID_i[test_index[0]],' Completed')
    
    testAccList.append(accuracy_lst)
    
    msg = "Completed combination #: {0:>2} of {1:>2}"
    print(msg.format((combo+1), len(combos)))

# for results use combos and test_data_id as legend and testAccList for accuracy results
    
#%% 
SID_list = []
max_1 = []
max_2 = []
max_3 = []
max_4 = []
max_5 = []
max_6 = []
max_7 = []
for SID in SID_i:
    SID_data = SID_dict[SID]
    max_1.append(max(max(SID_data['FSR1_L']),max(SID_data['FSR1_R'])))
    max_2.append(max(max(SID_data['FSR2_L']),max(SID_data['FSR2_R'])))
    max_3.append(max(max(SID_data['FSR3_L']),max(SID_data['FSR3_R'])))
    max_4.append(max(max(SID_data['FSR4_L']),max(SID_data['FSR4_R'])))
    max_5.append(max(max(SID_data['FSR5_L']),max(SID_data['FSR5_R'])))
    max_6.append(max(max(SID_data['FSR6_L']),max(SID_data['FSR6_R'])))
    max_7.append(max(max(SID_data['FSR7_L']),max(SID_data['FSR7_R'])))
    SID_list.append(SID)

max_vals = pd.DataFrame(SID_list,max_1)
    
