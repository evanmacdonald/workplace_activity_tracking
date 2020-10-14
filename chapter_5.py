# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:27:25 2019

@author: evanm
"""

import pandas as pd
import numpy as np
import seaborn as sn
import statistics as st
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

#%% import data and run basic correlation
comp = 'evanm'
model = 'svm'

file = open('C:/Users/'+comp+'/OneDrive/SFU/Thesis Document/Analysis Spreadsheets/RawData_perDay_DF_comp_V5.csv')
RawData = pd.read_csv(file)

#revise data to be in format for correlation
#select variables to include
SelectVars = ['sub_ID','day_num','good_data','age','weight','height','gender','dom_foot','insole_size','shoe_size',
              'sit_'+model,'stand_'+model,'walk_'+model,'sit_est','stand_est','walk_est','typical','pm_activity',
              'eq5d_mobility','eq5d_activities','eq5d_anxiety','eq5d_pain','eq5d_health',
              'sit_0-low_qtile_'+model,'sit_low_qtile-high_qtile_'+model,'sit_over_high_qtile_'+model,
              'stand_0-low_qtile_'+model,'stand_low_qtile-high_qtile_'+model,'stand_over_high_qtile_'+model,
              'walk_0-low_qtile_'+model,'walk_low_qtile-high_qtile_'+model,'walk_over_high_qtile_'+model]

CorrData = RawData[SelectVars]

#drop rows with bad/no data
null_index = CorrData[CorrData.good_data == 0].index
CorrData = CorrData.drop(null_index)

#use only first day of data (s.b. 29 participant's data)
#null_index = CorrData[CorrData.day_num != 1].index
#CorrData = CorrData.drop(null_index)

#this is the 'max pain' score
#CorrData['fadi_max'] = RawData[['fadi_23','fadi_24','fadi_25','fadi_26']].max(axis=1)
CorrData['pain'] = RawData['pain']
#change gender to form --> M=0, F=1
CorrData['gender'] = CorrData['gender'].apply(lambda x: 0 if x == 'M' else 1)
#change dom_foot to form --> R=0, L=1
CorrData['dom_foot'] = CorrData['dom_foot'].apply(lambda x: 0 if x == 'R' else 1)
#change typ to form --> Y=1, N=0
CorrData['typical'] = CorrData['typical'].apply(lambda x: 1 if x == 'Y' else 0)

#rearrange into desired order
VarsOrder = ['sub_ID','day_num','age','weight','height','gender','dom_foot','insole_size','shoe_size',
              'sit_'+model,'stand_'+model,'walk_'+model,'sit_est','stand_est','walk_est','typical','pm_activity',
              'sit_0-low_qtile_'+model,'sit_low_qtile-high_qtile_'+model,'sit_over_high_qtile_'+model,
              'stand_0-low_qtile_'+model,'stand_low_qtile-high_qtile_'+model,'stand_over_high_qtile_'+model,
              'walk_0-low_qtile_'+model,'walk_low_qtile-high_qtile_'+model,'walk_over_high_qtile_'+model,
              'eq5d_mobility','eq5d_activities','eq5d_anxiety','eq5d_pain','eq5d_health','pain']
col_names = ['SID','day_num','age','weight','height','gender','dom_foot','insole_size','shoe_size',
              'sit_act','stand_act','walk_act','sit_est','stand_est','walk_est','typical','pm_activity',
              'sit_0-low_qtile','sit_low_qtile-high_qtile','sit_over_high_qtile',
              'stand_0-low_qtile','stand_low_qtile-high_qtile','stand_over_high_qtile',
              'walk_0-low_qtile','walk_low_qtile-high_qtile','walk_over_high_qtile',
              'eq5d_mobility','eq5d_activities','eq5d_anxiety','eq5d_pain','eq5d_health','pain']
CorrData = CorrData[VarsOrder]
CorrData.columns = col_names

#%% create additional features
CorrData['day_duration'] = CorrData['sit_act']+CorrData['stand_act']+CorrData['walk_act']
CorrData['perc_on_feet'] = (CorrData['stand_act']+CorrData['walk_act'])/(CorrData['day_duration'])
CorrData['change_activity'] = CorrData['sit_0-low_qtile']+CorrData['sit_low_qtile-high_qtile']+CorrData['sit_over_high_qtile']+CorrData['stand_0-low_qtile']+CorrData['stand_low_qtile-high_qtile']+CorrData['stand_over_high_qtile']+CorrData['walk_0-low_qtile']+CorrData['walk_low_qtile-high_qtile']+CorrData['walk_over_high_qtile']
CorrData['weight_bearing'] = CorrData['stand_act']+CorrData['walk_act']
CorrData['bmi'] = CorrData['weight']*0.453592 / ((CorrData['height']*0.01)*(CorrData['height']*0.01))

#%% get results using method 1 - average of all days
#
#Data = pd.DataFrame(np.nan, index=np.arange(len(np.unique(CorrData['SID']))), columns=list(CorrData.drop(['typical', 'insole_size'], axis=1)),dtype=float)
#Data['SID'] = np.unique(CorrData['SID'])
#for SID in Data['SID']:
#    loc_index = Data[Data.SID==SID].index[0]
#    SID_data = CorrData[CorrData.SID == SID].drop(['typical', 'insole_size'], axis=1)
#    Data.loc[loc_index,1:] = SID_data.mean()
#    Data.loc[loc_index,'pain'] = SID_data['pain'].max()
#    Data.loc[loc_index,'day_num'] = SID_data['day_num'].max()
#
#Data = Data.drop(['weight','height','sit_est','stand_est','walk_est','pm_activity','eq5d_mobility','eq5d_activities','eq5d_anxiety','eq5d_pain','eq5d_health','perc_on_feet'],axis=1)

#%% get results using method 2 - day with maximum weightbearing

Data = pd.DataFrame(np.nan, index=np.arange(len(np.unique(CorrData['SID']))), columns=list(CorrData.drop(['typical', 'insole_size'], axis=1)),dtype=float)
Data['SID'] = np.unique(CorrData['SID'])
for SID in Data['SID']:
    loc_index = Data[Data.SID==SID].index[0]
    SID_data = CorrData[CorrData.SID == SID].drop(['typical', 'insole_size'], axis=1)
    max_index = SID_data[SID_data.weight_bearing==max(SID_data['weight_bearing'])].index[0]
    Data.loc[loc_index,:] = SID_data.loc[max_index,:]

Data = Data.drop(['weight','height','sit_est','stand_est','walk_est','pm_activity','eq5d_mobility','eq5d_activities','eq5d_anxiety','eq5d_pain','eq5d_health','perc_on_feet'],axis=1)

#%% Get univariate stats using t-test and fishers exact test
pain_index = Data[Data.pain == 1].index
no_pain_data = Data.drop(pain_index)
pain_data = Data.loc[pain_index,:]

T_results = pd.DataFrame(np.nan, index = np.arange(len(list(Data.drop('SID',axis=1)))), columns=['factor','pain_mean','pain_std','nopain_mean','nopain_std','p'])
T_results['factor'] = list(Data.drop('SID',axis=1))
for var in T_results['factor']:
    var_index = T_results[T_results.factor==var].index[0]
    T,p_t = ss.ttest_ind(pain_data[var],no_pain_data[var])
    Mean = [st.mean(pain_data[var]),st.mean(no_pain_data[var])]
    STD = [st.stdev(pain_data[var]),st.stdev(no_pain_data[var])]
    T_results.loc[var_index,'pain_mean'] = Mean[0]
    T_results.loc[var_index,'pain_std'] = STD[0]
    T_results.loc[var_index,'nopain_mean'] = Mean[1]
    T_results.loc[var_index,'nopain_std'] = STD[1]
    T_results.loc[var_index,'p'] = p_t


## Fishers exact test of independence (two nominal variables with sample size <1000)
## only needed for gender, dom_foot, and typical
#var = 'typical'
#n1 = len(pain_data)
#n2 = len(no_pain_data)
## [1-P,1-NP],[0-P,0-NP]
##OR, p_f = ss.fisher_exact([[12,82],[20,293]])
#OR, p_f = ss.fisher_exact([[sum(pain_data[var]),sum(no_pain_data[var])],[n1-sum(pain_data[var]),n2-sum(no_pain_data[var])]])
#
#print([[sum(pain_data[var]),sum(no_pain_data[var])],[n1-sum(pain_data[var]),n2-sum(no_pain_data[var])]])
#print(p_f)


#%% nice looking boxplot showing standing and weightbearing for figure 2
## using factor as the x-axis
#sitDF = pd.DataFrame(CorrData[['sit_act','pain']])
#sitDF['Factor'] = 'Sitting'
#sitDF.columns = ['time','pain','Factor']
#sitDF['pain'] = sitDF['pain'].replace([0],'No Pain')
#sitDF['pain'] = sitDF['pain'].replace([1],'Pain')
#standDF = pd.DataFrame(CorrData[['stand_act','pain']])
#standDF['Factor'] = 'Standing'
#standDF.columns = ['time','pain','Factor']
#standDF['pain'] = standDF['pain'].replace([0],'No Pain')
#standDF['pain'] = standDF['pain'].replace([1],'Pain')
#wbrDF = pd.DataFrame(CorrData[['walk_act','pain']])
#wbrDF['Factor'] = 'Walking'
#wbrDF.columns = ['time','pain','Factor']
#wbrDF['pain'] = wbrDF['pain'].replace([0],'No Pain')
#wbrDF['pain'] = wbrDF['pain'].replace([1],'Pain')
#cdf = pd.concat([sitDF,standDF,wbrDF])
#
#sitDF['pain'] = sitDF['pain'].replace('No Pain','No Pain Datapoint (n=53)')
#sitDF['pain'] = sitDF['pain'].replace('Pain','Pain Datapoint (n=39)')
#standDF['pain'] = standDF['pain'].replace('No Pain','No Pain Datapoint (n=53)')
#standDF['pain'] = standDF['pain'].replace('Pain','Pain Datapoint (n=39)')
#wbrDF['pain'] = wbrDF['pain'].replace('No Pain','No Pain Datapoint (n=53)')
#wbrDF['pain'] = wbrDF['pain'].replace('Pain','Pain Datapoint (n=39)')
#cdf_data = pd.concat([sitDF,standDF,wbrDF])
#
#fig3, ax3 = plt.subplots(figsize=(12,9))
##sn.stripplot(x='pain',y='time', hue='Factor',data=cdf, jitter=True, palette='Set2', split=True, linewidth=1, edgecolor='gray')
#sn.swarmplot(x='Factor',y='time', hue='pain',data=cdf_data, dodge=True, palette='Set2',linewidth=1,edgecolor='gray')
#sn.boxplot(x='Factor',y='time',hue='pain',data=cdf,notch=True,palette='Set2', fliersize=0)
#ax3.xaxis.set_tick_params(labelsize=15)
#ax3.yaxis.set_tick_params(labelsize=13)
#ax3.set_title('Foot Pain vs. Time Sitting, Standing and Walking',fontsize=19, pad=20)
#ax3.set_xlabel('', fontsize=15, labelpad=20)
#ax3.set_ylabel('Time (minutes)',fontsize=15, labelpad=20)
#ax3.legend(fontsize=13)
##filename = 'C:/Users/'+comp+'/sfuvault/Thesis/Results/Boxplots/'+factor+'.png'
##plt.savefig(filename)

#%% Normalize CorrData

def normalize(df):
    result = df.copy()
    for feature_name in df.drop(['SID'],axis=1).columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#normalize CorrData
NormData=normalize(Data)


#%% Get p-values for each factor using multiple logistic regression and a likelihood ratio test

#select_factors = ['height','gender','stand_act','walk_act',
#                  'weight_bearing','day_duration', 'bmi',
#                  'stand_0-low_qtile','stand_over_high_qtile',
#                  'walk_0-low_qtile','walk_over_high_qtile']

full_model = [#'gender',
              'stand_act',
#              'walk_act',
#              'bmi',
#              'day_duration',
#              'stand_0-low_qtile',
              'stand_over_high_qtile',
#              'walk_0-low_qtile',
#              'walk_over_high_qtile',
#              'change_activity'
              ]

reduced_model = [
              'stand_act',
#              'stand_over_high_qtile',
#              'walk_0-low_qtile'
                  ]

X_full = sm.add_constant(NormData[full_model])
#X['Intercept'] = 1
y = Data['pain'] #needs to be binary (0 or 1)

#set up logistical regression model and get stats
MLR_stats_full = sm.Logit(y,X_full).fit()
#show summary of model
print(MLR_stats_full.summary())

# get results of reduced model
X_red = sm.add_constant(NormData[reduced_model])
MLR_stats_red = sm.Logit(y,X_red).fit()

##Confidence Interval
#CI = MLR_stats.conf_int()
##Odds Ratio
#OR = np.exp(MLR_stats.params)
##coefficients
#coef = MLR_stats.params
##pvalues
#p = MLR_stats.pvalues

#results
result_stats_full = MLR_stats_full.conf_int(alpha=0.05)
result_stats_full['coef'] = MLR_stats_full.params
result_stats_full['p'] = MLR_stats_full.pvalues
result_stats_full.columns = ['2.5%', '97.5%', 'coef','p']
result_stats_full = result_stats_full[['coef','p']]

result_stats_red = MLR_stats_red.conf_int(alpha=0.05)
result_stats_red['coef'] = MLR_stats_red.params
result_stats_red['p'] = MLR_stats_red.pvalues
result_stats_red.columns = ['2.5%', '97.5%', 'coef','p']
result_stats_red = result_stats_red[['coef','p']]
result_stats_red['coef_change'] = 0

for factor in reduced_model:
    v1 = result_stats_full['coef'].loc[factor]
    v2 = result_stats_red['coef'].loc[factor]
    result_stats_red['coef_change'].loc[factor] = (v2-v1)/v1*100
##Marginal effects
#mfx = MLR_stats.get_margeff()
#print(mfx.summary())

#%% check linearity of logit response
factor = 'stand_act'
low_qtile = NormData[factor].quantile(0.25)
mid_qtile = NormData[factor].quantile(0.5)
high_qtile = NormData[factor].quantile(0.75)

NormData['high']=0
NormData['mid_low']=0
NormData['mid_high']=0

for i in range(0,(len(Data)-1)):
    cur_val = NormData[factor].loc[i]
    if cur_val <= mid_qtile and cur_val > low_qtile:
        NormData['mid_low'].loc[i]=1
    if cur_val <= high_qtile and cur_val > mid_qtile:
        NormData['mid_high'].loc[i]=1
    if cur_val > high_qtile:
        NormData['high'].loc[i]=1
    
reduced_model = ['mid_low','mid_high','high','stand_act', 'day_duration'] 

X = sm.add_constant(NormData[reduced_model])
y = NormData['pain'] #needs to be binary (0 or 1)

#set up logistical regression model and get stats
MLR_stats = sm.Logit(y,X).fit()

A = [((mid_qtile+low_qtile)/2),MLR_stats.params[1]]
B = [((high_qtile+mid_qtile)/2),MLR_stats.params[2]]
C = [((1+high_qtile)/2),MLR_stats.params[3]]

res = pd.DataFrame([A,B,C],columns=['x','y'])

plt.plot(res['x'],res['y'])
plt.title(factor)

#%% OBSOLETE METHOD Get p-values for each factor using simple logistig regression and a likelihood ratio test

#def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=None):
#    """
#    Compute the likelihood ratio test for a model trained on the set of features in
#    `features_alternate` vs a null model using McFadden's Pseudo R^2 approach.
#    the null model simply uses the class probabilities.
#    Returns the p-value, which can be used to accept or reject the null hypothesis.
#    """
#    labels = np.array(labels)
#    features_alternate = np.array(features_alternate)
#    
#    null_prob = sum(labels) / float(labels.shape[0]) * np.ones(labels.shape)
#    df = features_alternate.shape[1]
#    
#    lr_model.fit(features_alternate, labels)
#    alt_prob = lr_model.predict_proba(features_alternate)
#
#    alt_log_likelihood = -log_loss(labels,
#                                   alt_prob,
#                                   normalize=False)
#    null_log_likelihood = -log_loss(labels,
#                                    null_prob,
#                                    normalize=False)
#    R = (null_log_likelihood-alt_log_likelihood)/null_log_likelihood
#    LR = 2 * (alt_log_likelihood - null_log_likelihood)
#    p_value = chi2.sf(LR, df)
#
#    return p_value, R, LR
#
#select_factors = ['age','weight','height','gender','sit_act','stand_act','walk_act','typical','pm_activity',
#              'sit_0-60s','sit_60-300s','sit_over_300s',
#              'stand_0-60s','stand_60-300s','stand_over_300s',
#              'walk_0-60s','walk_60-300s','walk_over_300s']
#
#'''Null Hupothesis: There is no relation between 'factor' and 'pain'''
#p = []
#r = []
#OR = []
#LR = []
#factor_name = []
#LogRegClf = LogisticRegression(fit_intercept = True, C = 1e9)
#
#for i in range(0,len(select_factors)):
#    factor = select_factors[i]
#    X = NormCorrData[factor].values.reshape(-1,1)
#    y = CorrData['pain']
#    p_val,r_val,LR_val = likelihood_ratio_test(X,y,LogRegClf)
#    OR_val = np.exp(LogRegClf.coef_[0][0])
#    p.append(p_val)
#    r.append(r_val)
#    OR.append(OR_val)
#    LR.append(LR_val)
#    factor_name.append(factor)
#
#stats_results = pd.DataFrame({'Factor_Name':factor_name, 'Odds_Rato':OR, 'p_value':p, 'r_value':r, 'Likelihood-Ratio':LR})
#
#sig_index = stats_results[stats_results.p_value < 0.001].index
#sig_factors = stats_results.loc[sig_index,:]