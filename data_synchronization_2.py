# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:12:17 2018

@author: evanm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Kintec_Functions import read_binary

def combined_data_two_insole():
    column_names = ['time', 'FSR1_R', 'FSR2_R', 'FSR3_R', 'FSR4_R', 
                    'FSR5_R', 'FSR6_R', 'FSR7_R', 'X_R', 'Y_R', 'Z_R', 'ActivityState_R',
                    'time_L', 'FSR1_L', 'FSR2_L', 'FSR3_L', 'FSR4_L', 
                    'FSR5_L', 'FSR6_L', 'FSR7_L', 'X_L', 'Y_L', 'Z_L', 'ActivityState']
    
    
    # Use this building block for data without solutions (data collected from a full day)
    right = 'C:/Users/Evan Macdonald/sfuvault/Thesis/Trial Data/Full Trial/SID10/Calibration/260926R10.bin'
    data_right = read_binary(right)
    left = 'C:/Users/Evan Macdonald/sfuvault/Thesis/Trial Data/Full Trial/SID10/Calibration/260926L10.bin'
    data_left = read_binary(left)
    #use if data from one insole dows not exist
#    data_right = data_left.copy(deep=True)
#    data_right[:] = 0
    # Set offset times based on video data
    ts_data_right = 0 #seconds elapsed at start of data recording
    ts_data_left = 5.768 #seconds elapsed at start of data recording
    #(sol_L[0]-(((sol_R[1]-sol_R[0])/2)+sol_R[0]))/45
    # Trim data so they are synchronized and the same length
    if ts_data_right > ts_data_left:
        data_left = data_left[int(ts_data_right*45.45):]
        if len(data_left) < len(data_right):
            data_right = data_right[0:len(data_left)]
        elif len(data_left) > len(data_right):
            data_left = data_left[0:len(data_right)]
    elif ts_data_right < ts_data_left:
        data_right = data_right[int(ts_data_left*45.45):]    
        if len(data_left) < len(data_right):
            data_right = data_right[0:len(data_left)]
        elif len(data_left) > len(data_right):
            data_left = data_left[0:len(data_right)]
    #reset indexes
    data_right = data_right.reset_index(drop=True)
    data_left = data_left.reset_index(drop=True)
    data = pd.concat([data_right, data_left], axis=1)
    data.columns=column_names
    data['ActivityState']=np.zeros(len(data)) #sets activity state to zero since we dont know solution
    #drop duplicate columns (time and activity state)
    data = data.drop(['time_L','ActivityState_R'], axis=1)
    #arrange data so all FSR data is first, then accelerometer data
    arranged_column_names = ['time', 'FSR1_R', 'FSR2_R', 'FSR3_R', 'FSR4_R', 'FSR5_R', 'FSR6_R', 'FSR7_R', 
                             'FSR1_L', 'FSR2_L', 'FSR3_L', 'FSR4_L', 'FSR5_L', 'FSR6_L', 'FSR7_L', 
                             'X_R', 'Y_R', 'Z_R', 'X_L', 'Y_L', 'Z_L','ActivityState']
    data = data[arranged_column_names]
    SynchronizedData = data
    
    #use to crop of start of data
#    SynchronizedData = SynchronizedData[35877:]
#    SynchronizedData = SynchronizedData.reset_index(drop=True)
    
    # Select the data to use for training and testing
    test_data = SynchronizedData
    train_data = SynchronizedData
    
    return train_data, test_data

#%%
_, test_data = combined_data_two_insole()

threshold = 300
dist = 50000
crosslocs = np.zeros(dist)
for i in range(1,dist):
    curval = test_data['FSR7_R'][i-1]
    nextval = test_data['FSR7_R'][i]
    if curval<threshold and nextval>=threshold:
        crosslocs[i] = 400
crosslocs_R = np.where(crosslocs==400)[0]
crossvals_R = test_data['FSR7_R'][crosslocs_R]

crosslocs = np.zeros(dist)
for i in range(1,dist):
    curval = test_data['FSR7_L'][i-1]
    nextval = test_data['FSR7_L'][i]
    if curval<threshold and nextval>=threshold:
        crosslocs[i] = 400
crosslocs_L = np.where(crosslocs==400)[0]
crossvals_L = test_data['FSR7_L'][crosslocs_L]

#%%
#distance between threshold crossings
diffs_R = np.asarray([y - x for x,y in zip(crosslocs_R,crosslocs_R[1:])])
j = 0
sol_R = []
for i in range(1,len(diffs_R)):
    curval = diffs_R[i]
    if curval>35 and curval<65:
        j = j+1
        if j>5:
            sol_R = np.append(sol_R,crosslocs_R[i-2])
    else:
        j=0

diffs_L = np.asarray([y - x for x,y in zip(crosslocs_L,crosslocs_L[1:])])
j = 0
sol_L = []
for i in range(1,len(diffs_L)):
    curval = diffs_L[i]
    if curval>35 and curval<65:
        j = j+1
        if j>5:
            sol_L = np.append(sol_L,crosslocs_L[i-2])
    else:
        j=0

#%%
cut = 50000
#cut = len(test_data)
fig,ax=plt.subplots(sharex=True, nrows=2, ncols=1)
ax[0].plot(test_data['FSR1_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR2_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR3_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR4_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR5_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR6_R'][:cut], linewidth=0.5)
ax[0].plot(test_data['FSR7_R'][:cut], linewidth=0.5)
ax[0].plot(crosslocs_R,crossvals_R, 'ro',label='Crossing')
#ax[0].plot(sol_R,np.ones(len(sol_R))*500,'bo')
ax[0].set_ylim((0,900))
ax[0].legend(loc='center left',prop={'size':16})
ax[0].set_title('FSR data Right',fontsize=25)
ax[0].set_ylabel('FSR Output',fontsize=22)
ax[0].tick_params(axis='both', which='major', labelsize=18)

ax[1].plot(test_data['FSR1_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR2_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR3_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR4_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR5_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR6_L'][:cut], linewidth=0.5)
ax[1].plot(test_data['FSR7_L'][:cut], linewidth=0.5)
ax[1].plot(crosslocs_L,crossvals_L, 'ro',label='Crossing')
#ax[1].plot(sol_L,np.ones(len(sol_L))*500,'bo')
ax[1].set_ylim((0,900))
ax[1].legend(loc='center left',prop={'size':16})
ax[1].set_title('FSR Data Left',fontsize=25)
ax[1].set_ylabel('FSR Output',fontsize=22)
ax[1].tick_params(axis='both', which='major', labelsize=18)

mng=plt.get_current_fig_manager() 
mng.window.showMaximized() #maximize figure 
plt.show()

#%%
#(sol_L[0]-(((sol_R[1]-sol_R[0])/2)+sol_R[0]))/45
