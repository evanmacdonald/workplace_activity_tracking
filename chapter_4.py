# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:57:16 2019

@author: evanm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

comp = 'evanm'
model = 'svm'

#%%

file = open('C:/Users/'+comp+'/OneDrive/SFU/Thesis Document/Analysis Spreadsheets/RawData_perDay_DF_comp_V5.csv')
RawData = pd.read_csv(file)

SelectVars = ['sub_ID','day_num','good_data','age','weight','height','gender',
              'sit_'+model,'stand_'+model,'walk_'+model,'sit_est','stand_est','walk_est','pain',
              'sit_0-low_qtile_'+model,'sit_low_qtile-high_qtile_'+model,'sit_over_high_qtile_'+model,
              'stand_0-low_qtile_'+model,'stand_low_qtile-high_qtile_'+model,'stand_over_high_qtile_'+model,
              'walk_0-low_qtile_'+model,'walk_low_qtile-high_qtile_'+model,'walk_over_high_qtile_'+model]

Data = RawData[SelectVars]

#drop rows with bad/no data
Data = Data.drop(Data[Data.good_data == 0].index)

#change gender to form --> M=0, F=1
Data['gender'] = Data['gender'].apply(lambda x: 0 if x == 'M' else 1)

#rearrange into desired order
VarsOrder = ['sub_ID','day_num','good_data','age','weight','height','gender',
              'sit_'+model,'stand_'+model,'walk_'+model,'sit_est','stand_est','walk_est','pain',
              'sit_0-low_qtile_'+model,'sit_low_qtile-high_qtile_'+model,'sit_over_high_qtile_'+model,
              'stand_0-low_qtile_'+model,'stand_low_qtile-high_qtile_'+model,'stand_over_high_qtile_'+model,
              'walk_0-low_qtile_'+model,'walk_low_qtile-high_qtile_'+model,'walk_over_high_qtile_'+model]
col_names = ['SID','day_num','good_data','age','weight','height','gender',
              'sit_act','stand_act','walk_act','sit_est','stand_est','walk_est','pain',
              'sit_0-low_qtile','sit_low_qtile-high_qtile','sit_over_high_qtile',
              'stand_0-low_qtile','stand_low_qtile-high_qtile','stand_over_high_qtile',
              'walk_0-low_qtile','walk_low_qtile-high_qtile','walk_over_high_qtile']

Data = Data[VarsOrder]
Data.columns = col_names
Data = Data.reset_index(drop=True)

#Additional Features
Data['day_duration'] = Data['sit_act']+Data['stand_act']+Data['walk_act']
Data['perc_on_feet'] = ((Data['stand_act']+Data['walk_act'])/(Data['day_duration']))*100
Data['perc_walk'] = (Data['walk_act']/Data['day_duration'])*100
Data['perc_stand'] = (Data['stand_act']/Data['day_duration'])*100
Data['perc_sit'] = (Data['sit_act']/Data['day_duration'])*100
Data['change_activity'] = Data['sit_0-low_qtile']+Data['sit_low_qtile-high_qtile']+Data['sit_over_high_qtile']+Data['stand_0-low_qtile']+Data['stand_low_qtile-high_qtile']+Data['stand_over_high_qtile']+Data['walk_0-low_qtile']+Data['walk_low_qtile-high_qtile']+Data['walk_over_high_qtile']
Data['weight_bearing'] = Data['stand_act']+Data['walk_act']
Data['bmi'] = Data['weight']*0.453592 / ((Data['height']*0.01)*(Data['height']*0.01))

#%% get results

results_labels = ['SID','num_days','sit_mean','sit_std','stand_mean','stand_std','walk_mean','walk_std','wbr_mean','wbr_std','act_changes_mean','act_changes_std','bmi']
results = pd.DataFrame(np.nan, index=np.arange(len(np.unique(Data['SID']))), columns=results_labels,dtype=float)
results['SID'] = np.unique(Data['SID'])
for SID in Data['SID']:
    loc_index = results[results.SID==SID].index[0]
    SID_data = Data[Data.SID == SID]
    results.loc[loc_index,'num_days'] = len(SID_data)
    results.loc[loc_index,'sit_mean'] = np.mean(SID_data['perc_sit'])
    results.loc[loc_index,'sit_std'] = np.std(SID_data['perc_sit'])
    results.loc[loc_index,'stand_mean'] = np.mean(SID_data['perc_stand'])
    results.loc[loc_index,'stand_std'] = np.std(SID_data['perc_stand'])
    results.loc[loc_index,'walk_mean'] = np.mean(SID_data['perc_walk'])
    results.loc[loc_index,'walk_std'] = np.std(SID_data['perc_walk'])
    results.loc[loc_index,'wbr_mean'] = np.mean(SID_data['perc_on_feet'])
    results.loc[loc_index,'wbr_std'] = np.std(SID_data['perc_on_feet'])
    results.loc[loc_index,'act_changes_mean'] = np.mean(SID_data['change_activity'])
    results.loc[loc_index,'act_changes_std'] = np.std(SID_data['change_activity'])

#%% plot overall activity data
plt_data = results.sort_values(by = 'sit_mean')
plt_data = plt_data.drop(plt_data[plt_data.num_days==1].index)
x = np.arange(0,len(plt_data))
plt.plot(x,plt_data['sit_mean'],label='Sitting')
plt.plot(x,plt_data['stand_mean'],label='Standing')
plt.plot(x,plt_data['walk_mean'],label='Walking')
plt.fill_between(x,(plt_data['sit_mean']-plt_data['sit_std']),(plt_data['sit_mean']+plt_data['sit_std']),alpha=0.2)
plt.fill_between(x,(plt_data['stand_mean']-plt_data['stand_std']),(plt_data['stand_mean']+plt_data['stand_std']),alpha=0.2)
plt.fill_between(x,(plt_data['walk_mean']-plt_data['walk_std']),(plt_data['walk_mean']+plt_data['walk_std']),alpha=0.2)
plt.legend(fontsize=15)
plt.title('Activity Breakdown for Each Participant',fontsize=20)
plt.xlabel('Participant Number',fontsize=15)
plt.ylabel('Average Percent of Workday (%)',fontsize=15)

#%% activity transition graph
fig,ax1 = plt.subplots()
ax1.errorbar(x,plt_data['act_changes_mean'],plt_data['act_changes_std'],marker='.',mfc='r',mec='r',linestyle='',
             ecolor='k',elinewidth=1, capsize=1.5, label='Number of Activity Changes')
ax1.legend(loc=3)
plt.xlabel('Participant Number')
plt.title('Number of Activity Changes and % of Workday Spent on Feet')
ax1.set_ylabel('Number of Activity Changes')

ax2 = ax1.twinx()
ax2.plot(x,plt_data['wbr_mean'], label = '% of Workday on Feet')
ax2.legend(loc=1)
ax2.set_ylabel('Perfect of Workday on Feet (%)')

