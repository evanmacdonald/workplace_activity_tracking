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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#%% import all data
# NOTE: SID21,22,23 do not have calibration data
# NOTE: SID18 and SID27 should be left out as it's data causes issues for the algorithm
print('Importing Data...')

# Declared Variables
order = 2
cutoff = 10
buffLen = 40
numvars = 16
decimation = 10
red_buff = int(buffLen/decimation)

comp = 'Evan Macdonald'

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID01/SynchronizedData/SID01_Calibration.csv')
SID01 = pd.read_csv(file)
SID01 = kf.downsample(SID01,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID02/SynchronizedData/SID02_Calibration.csv')
SID02 = pd.read_csv(file)
SID02 = kf.downsample(SID02,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID03/SynchronizedData/SID03_Calibration.csv')
SID03 = pd.read_csv(file)
SID03 = kf.downsample(SID03,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID04/SynchronizedData/SID04_Calibration.csv')
SID04 = pd.read_csv(file)
SID04 = kf.downsample(SID04,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID05/SynchronizedData/SID05_Calibration.csv')
SID05 = pd.read_csv(file)
SID05 = kf.downsample(SID05,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID06/SynchronizedData/SID06_Calibration.csv')
SID06 = pd.read_csv(file)
SID06 = kf.downsample(SID06,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID07/SynchronizedData/SID07_Calibration.csv')
SID07 = pd.read_csv(file)
SID07 = kf.downsample(SID07,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID08/SynchronizedData/SID08_Calibration.csv')
SID08 = pd.read_csv(file)
SID08 = kf.downsample(SID08,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID09/SynchronizedData/SID09_Calibration.csv')
SID09 = pd.read_csv(file)
SID09 = kf.downsample(SID09,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID10/SynchronizedData/SID10_Calibration.csv')
SID10 = pd.read_csv(file)
SID10 = kf.downsample(SID10,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID11/SynchronizedData/SID11_Calibration.csv')
SID11 = pd.read_csv(file)
SID11 = kf.downsample(SID11,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID12/SynchronizedData/SID12_Calibration.csv')
SID12 = pd.read_csv(file)
SID12 = kf.downsample(SID12,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID13/SynchronizedData/SID13_Calibration.csv')
SID13 = pd.read_csv(file)
SID13 = kf.downsample(SID13,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID14/SynchronizedData/SID14_Calibration.csv')
SID14 = pd.read_csv(file)
SID14 = kf.downsample(SID14,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID15/SynchronizedData/SID15_Calibration.csv')
SID15 = pd.read_csv(file)
SID15 = kf.downsample(SID15,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID16/SynchronizedData/SID16_Calibration.csv')
SID16 = pd.read_csv(file)
SID16 = kf.downsample(SID16,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID17/SynchronizedData/SID17_Calibration.csv')
SID17 = pd.read_csv(file)
SID17 = kf.downsample(SID17,decimation)

#leave this one out, it is not good data
file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID18/SynchronizedData/SID18_Calibration.csv')
SID18 = pd.read_csv(file)
SID18 = kf.downsample(SID18,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID19/SynchronizedData/SID19_Calibration.csv')
SID19 = pd.read_csv(file)
SID19 = kf.downsample(SID19,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID20/SynchronizedData/SID20_Calibration.csv')
SID20 = pd.read_csv(file)
SID20 = kf.downsample(SID20,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID24/SynchronizedData/SID24_Calibration.csv')
SID24 = pd.read_csv(file)
SID24 = kf.downsample(SID24,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID25/SynchronizedData/SID25_Calibration.csv')
SID25 = pd.read_csv(file)
SID25 = kf.downsample(SID25,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID26/SynchronizedData/SID26_Calibration.csv')
SID26 = pd.read_csv(file)
SID26 = kf.downsample(SID26,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID28/SynchronizedData/SID28_Calibration.csv')
SID28 = pd.read_csv(file)
SID28 = kf.downsample(SID28,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID29/SynchronizedData/SID29_Calibration.csv')
SID29 = pd.read_csv(file)
SID29 = kf.downsample(SID29,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID30/SynchronizedData/SID30_Calibration.csv')
SID30 = pd.read_csv(file)
SID30 = kf.downsample(SID30,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID31/SynchronizedData/SID31_Calibration.csv')
SID31 = pd.read_csv(file)
SID31 = kf.downsample(SID31,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID32/SynchronizedData/SID32_Calibration.csv')
SID32 = pd.read_csv(file)
SID32 = kf.downsample(SID32,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID33/SynchronizedData/SID33_Calibration.csv')
SID33 = pd.read_csv(file)
SID33 = kf.downsample(SID33,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID34/SynchronizedData/SID34_Calibration.csv')
SID34 = pd.read_csv(file)
SID34 = kf.downsample(SID34,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID35/SynchronizedData/SID35_Calibration.csv')
SID35 = pd.read_csv(file)
SID35 = kf.downsample(SID35,decimation)

file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID36/SynchronizedData/SID36_Calibration.csv')
SID36 = pd.read_csv(file)
SID36 = kf.downsample(SID36,decimation)

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


#%% Get predictions based on a Leave One Out cross validation
print('Getting solutions...')


#Lists for saving data
test_data_id = []
train_data_lst = []
prediction_lst = []
solution_lst = []
accuracy_lst = []

# Get results for each participant's data using leave one out cross validation.
# Each iteration selects a new dataset for testing and creates a training set with the rest fo the datasets 
loo = LeaveOneOut()
for train_index, test_index in loo.split(SID_i):
    # Reference lists
    test_data_id.append(SID_i[test_index[0]])
    train_data_lst.append(train_index)
    
    #get training and testing data
    test_data = SID_dict[SID_i[test_index[0]]]
    train_data = SID_dict[SID_i[train_index[0]]]
    for i in train_index[1:]:
        df = SID_dict[SID_i[i]]
        train_data = train_data.append(df).reset_index(drop=True)
    
    #process data
    X, Y = kf.pre_process_data(train_data,order,cutoff,red_buff,numvars)
    X_test, Y_test = kf.pre_process_data(test_data,order,cutoff,red_buff,numvars)
    select_feats = np.asarray([0,7,16,23, #FSR1
                           1,8,17,24, #FSR2
                           2,9,18,25, #FSR3
                           3,10,19,26, #FSR4
                           4,11,20,27, #FSR5
                           5,12,21,28, #FSR6
                           6,13,22,29, #FSR7
                           14,15,30,31 #Acc
                           ])
    
    
    X_select = np.take(X,select_feats,axis=1)
    X_test_select = np.take(X_test,select_feats,axis=1)
    
    #get results (select only one model, not both)
    model = svm.SVC(kernel='poly', gamma=0.05, C=105)
#    model = LogisticRegression(multi_class='multinomial',solver='newton-cg', C=5000, tol=0.01)
    model.fit(X_select,Y)
    y_pred = model.predict(X_test_select)
    accuracy = accuracy_score(Y_test,y_pred)
    accuracy_lst.append(accuracy)
    prediction_lst.append(y_pred)
    solution_lst.append(Y_test)  
    print('   ',SID_i[test_index[0]],' Completed')
        
print('Solutions obtained...')

# Get accuracy and CM for each validation step
for i in range(0,len(test_data_id)):
    test_data = SID_dict[test_data_id[i]]
    y_pred = prediction_lst[i]
    y_out_val = kf.reshape_ybar(y_pred,test_data)
    test_labels = np.asarray(test_data['ActivityState'],dtype='int32')
    title = test_data_id[i]
    
#
# Put accuracy next to the ID name in a dataframe for further analysis
results = pd.DataFrame({'Test Data ID':test_data_id, 'Accuracy':accuracy_lst})

#%% Plot results from specific Participant
    
## modify this to plot results for the participant you want
#ID = 'SID25'
#
#y_ = prediction_lst[test_data_id.index(ID)]
#test_labels = solution_lst[test_data_id.index(ID)]
#test_data = SID_dict[test_data_id[test_data_id.index(ID)]]
#y_out_val = kf.reshape_ybar(y_,test_data)
#y_out_val = np.insert(y_out_val,0,(np.asarray(np.ones(buffLen))))
#y_out_val = y_out_val[0:-40]
#
#kf.plot_results(test_data,y_out_val)
#x_ticks = np.arange(0,(len(test_data)))/45
#font = {'fontname':'Arial'}
#fig,ax=plt.subplots(figsize=(19,5))
#ax.plot(x_ticks,test_data['ActivityState'], '0.6', label='Actual', linewidth = 6)
#ax.plot(x_ticks,y_out_val, 'r', label='Predicted', linewidth = 1)
#ax.set_ylim((0.5,3.5))
#ax.legend(prop={'size':16})
#ax.set_title('Activity State vs Time', fontsize=22,pad=15,**font)
#ax.set_xlabel('Time (s)', fontsize=17, labelpad=20,**font)
#ax.set_ylabel('Activity State', fontsize=17, labelpad=20,**font)
#ax.yaxis.set_tick_params(labelsize=15)
#ax.xaxis.set_tick_params(labelsize=15)
#plt.yticks([1,2,3], ['Sit', 'Stand', 'Walk'])
#plt.tight_layout()
#plt.show()