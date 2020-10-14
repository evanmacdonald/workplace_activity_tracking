# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:48:46 2018

@author: Evan Macdonald
"""
import pandas as pd
import numpy as np
import struct
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from sklearn.metrics import confusion_matrix
#import cv2
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, freqz

# opens and imports data from output file from device up to Kintec_V1.1
def read_data(file_path):
    column_names = ['time', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 'FSR5', 'FSR6', 'FSR7', 'X', 'Y', 'Z', 'ActivityState']
    data = pd.read_csv(
            file_path, 
            header=None, 
            skiprows=2, 
            dtype={'FSR1':np.float32, 'FSR2':np.float32, 'FSR3':np.float32, 'FSR4':np.float32, 'FSR5':np.float32, 'FSR6':np.float32, 'FSR7':np.float32, 'X':np.float32, 'Y':np.float32, 'Z':np.float32}, 
            names = column_names)
    return data

# opens data from a binary file with all float values. 
# deletes any zero values (filler bytes)
# works with Kintec_V2 teensy software
# Note: importing all data as float, have cast it all to float in the Arduino code
# future improvement would be to import as multiple datatypes to get accurate timestamps
def read_binary(file_path):
    sample = ''
    fill = ''
    for i in range(0,372):
        sample = sample + 'I10f' 
    for i in range(0,16):
        fill = fill + 'x'
    struct_fmt = sample + fill
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    results = []
    with open(file_path, 'rb') as f:
        while True:
            data_chunk = f.read(struct_len)
            if not data_chunk: break
            s = struct_unpack(data_chunk)
            results.append(s)

    raw_data = np.asarray(results)
    raw_data = np.reshape(raw_data,(-1,11))
    time_int = np.asarray(raw_data[:,0], dtype=np.dtype('u4'))

    column_names = ['time', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 
                'FSR5', 'FSR6', 'FSR7', 'X', 'Y', 'Z']
    data = pd.DataFrame(raw_data, columns=column_names)
    data['time'] = time_int
    data['ActivityState'] = 0
    return data

def window(data, size, overlap):
    i = 0
    while i < data.count():
        yield int(i), int(i+size)
        i += (size/overlap) #equates to a 50% overlap of second half of data. was (size/2)


# breaks data into buffer segments       
def segment_values(data, window_size,numVars):
    segments = np.asarray(data)
    #cut to be a shape that can fit into N*window_size
    #note, takes data off the end of the array
    maxLen = (int(len(data)/window_size))*window_size
    segments = segments[:maxLen]
    #pull out labels and crop off timestamps etc.
    labels = segments[:,-1]
    segments = np.delete(segments,(0,(numVars+1)),1)
    #buffer data to the right shape
    segments = np.reshape(segments,(-1,window_size,numVars))
    labels = np.reshape(labels,(-1,window_size))
    labels = stats.mode(labels,axis=1)[0]
    labels = labels[:,0]
    return segments, labels

# Slow way of buffering databut can accomodate an overlap
def segment_values_2(data, window_size, overlap):
    numvars = 20
    segments = np.empty((0,window_size,numvars))
    labels = np.empty((0))
    for (start,end) in window(data['time'], window_size, (window_size/(window_size-overlap))):
        if(len(data['time'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack(
                    [data["FSR1_R"][start:end],
                     data["FSR2_R"][start:end],
                     data["FSR3_R"][start:end],
                     data["FSR4_R"][start:end],
                     data["FSR5_R"][start:end],
                     data["FSR6_R"][start:end],
                     data["FSR7_R"][start:end],
                     data["FSR1_L"][start:end],
                     data["FSR2_L"][start:end],
                     data["FSR3_L"][start:end],
                     data["FSR4_L"][start:end],
                     data["FSR5_L"][start:end],
                     data["FSR6_L"][start:end],
                     data["FSR7_L"][start:end],
                     data["X_R"][start:end],
                     data["Y_R"][start:end],
                     data["Z_R"][start:end],
                     data["X_L"][start:end],
                     data["Y_L"][start:end],
                     data["Z_L"][start:end]])])
            labels = np.append(labels, stats.mode(data['ActivityState'][start:end])[0][0])
    return segments, labels

# Very fast way of buffering data if using no overlap in the data
def segment_values_NO(data, window_size):
    segments = np.asarray(data)
    #cut to be a shape that can fit into N*window_size
    #note, takes data off the end of the array
    maxLen = (int(len(data)/window_size))*window_size
    segments = segments[:maxLen]
    #pull out labels and crop off timestamps etc.
    labels = segments[:,22]
    segments = np.delete(segments,(0,1,22),1)
    #buffer data to the right shape
    segments = np.reshape(segments,(-1,window_size,20))
    labels = np.reshape(labels,(-1,window_size))
    labels = stats.mode(labels,axis=1)[0]
    labels = labels[:,0]
    return segments, labels

def get_features(X):
    #average values
    mean = np.mean(X,axis=2)
    #Standard Deviation
    std = np.std(X,axis=2)
    features = np.concatenate((mean,std),axis=1)
    return features    

def get_max_features(X):
    '''
    Add in anything that is in the shape of [*,1,20]
    '''
    #average values
    mean = np.mean(X,axis=2)
    #Standard Deviation
    std = np.std(X,axis=2)
#    #Median
#    median = np.median(X,axis=2)
#    #Cumulative sum of sensor values across the buffer
#    cum_sum = np.sum(X,axis=2)
#    cum_sum = cum_sum/cum_sum.max()
#    #Absolute average discrete difference
#    difference = np.absolute(np.mean(np.diff(X,axis=2),axis=2))
#    difference = difference/difference.max()

#    features = np.concatenate((mean,std,median,cum_sum,difference),axis=1)
    features = np.concatenate((mean,std),axis=1)
    features = np.reshape(features,[-1,features.shape[1]*features.shape[2]]) 
#    '''
#    Add in anything that is in the shape of [*,1]
#    '''
#    
#    #sum of the mean values of the sensors within the buffer window
#    sum_means_R = np.reshape(np.sum(mean[:,0,0:7],axis=1),[-1,1])
#    sum_means_R = sum_means_R/sum_means_R.max()
#    sum_means_L = np.reshape(np.sum(mean[:,0,8:14],axis=1),[-1,1])
#    sum_means_L = sum_means_L/sum_means_L.max()
#    sum_means = sum_means_R + sum_means_L
#    sum_means = sum_means/sum_means.max()
#    
#    features = np.concatenate((features,sum_means_R,sum_means_L,sum_means), axis=1)
    return features

def low_pass_filter(X,order,cutoff):
    # order = order of filter
    fs = 45.45       # sample rate, Hz
    #cutoff = desired cutoff frequency of the filter, Hz
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b, a, X)
    return y
    
#input must be dataframe with labeled columns
def process_acc(X,order,cutoff):
    
    X = X.drop('Unnamed: 0',axis =1)
    header = np.asarray(X[:][0:cutoff])
    X['X_R'] = low_pass_filter(X['X_R'],order,cutoff)
    X['Y_R'] = low_pass_filter(X['Y_R'],order,cutoff)
    X['Z_R'] = low_pass_filter(X['Z_R'],order,cutoff)
    X['X_L'] = low_pass_filter(X['X_L'],order,cutoff)
    X['Y_L'] = low_pass_filter(X['Y_L'],order,cutoff)
    X['Z_L'] = low_pass_filter(X['Z_L'],order,cutoff)
    X[:][0:cutoff] = header
    acc_R = np.sqrt((X['X_R']**2)+(X['Y_R']**2)+(X['Z_R']**2))
    acc_L = np.sqrt((X['X_L']**2)+(X['Y_L']**2)+(X['Z_L']**2))
    X['acc_R'] = acc_R
    X['acc_L'] = acc_L
    X = X.drop(['X_R', 'Y_R', 'Z_R', 'X_L', 'Y_L', 'Z_L'],axis=1)
    X = X[['time', 'FSR1_R', 'FSR2_R', 'FSR3_R', 'FSR4_R', 'FSR5_R',
       'FSR6_R', 'FSR7_R', 'FSR1_L', 'FSR2_L', 'FSR3_L', 'FSR4_L', 'FSR5_L',
       'FSR6_L', 'FSR7_L', 'acc_R', 'acc_L','ActivityState']]
    return X

def normalize(X):
    out = np.zeros(X.shape)
    maxFSR = np.max(X[:,:,:,0:7])
    maxACC = np.max(X[:,:,:,7:10])
    out[:,:,:,0:7]=X[:,:,:,0:7]/maxFSR
    out[:,:,:,7:10]=X[:,:,:,7:10]/maxACC
    return out

def normalize_2(X):
    out = np.zeros(X.shape)
    maxFSR = np.max(X[:,:,:,0:14])
    maxACC = np.max(X[:,:,:,14:20])
    out[:,:,:,0:14]=X[:,:,:,0:14]/maxFSR
    out[:,:,:,14:20]=X[:,:,:,14:20]/maxACC
    return out

def normalize_3(X):
    #normalizes dataframe based on maximum FSR value and max ACC value (of all sensors)
    X = X.drop('Unnamed: 0',axis =1)
    out = pd.DataFrame(np.zeros(X.shape), columns=X.columns)
    out['time'] = X['time']
    out['ActivityState'] = X['ActivityState']
    max_vals = np.max(X)
    min_vals = np.min(X)
    max_FSR = np.max(max_vals[1:15])
    min_FSR = np.min(min_vals[1:15])
    max_ACC = np.max(max_vals[16:22])
    min_ACC = np.min(min_vals[16:22])
    out.iloc[:,1:15] = (X.iloc[:,1:15]-min_FSR)/(max_FSR-min_FSR)
    out.iloc[:,15:21] = (X.iloc[:,15:21]-min_ACC)/(max_ACC-min_ACC)
    return out

#use when using process_acc first
def normalize_4(X):
    #normalizes dataframe based on maximum FSR value and max ACC value (of all sensors)
    out = pd.DataFrame(np.zeros(X.shape), columns=X.columns)
    out['time'] = X['time']
    out['ActivityState'] = X['ActivityState']
    max_vals = np.max(X)
    min_vals = np.min(X)
    max_FSR = np.max(max_vals[1:15]) #815
    min_FSR = np.min(min_vals[1:15]) #1
    max_ACC = np.max(max_vals[15:17]) #67.865
    min_ACC = np.min(min_vals[15:17]) #0.534
    out.iloc[:,1:15] = (X.iloc[:,1:15]-min_FSR)/(max_FSR-min_FSR)
    out.iloc[:,15:17] = (X.iloc[:,15:17]-min_ACC)/(max_ACC-min_ACC)
    return out

def normalize_MLR(X):
    #normalizes dataframe based on maximum FSR value and max ACC value (of all sensors)
    out = pd.DataFrame(np.zeros(X.shape), columns=X.columns)
    out['time'] = X['time']
    out['ActivityState'] = X['ActivityState']
    max_FSR = 815 #maximum on entire training set
    min_FSR = 1
    max_ACC = 67.865
    min_ACC = 0.534
    out.iloc[:,1:15] = (X.iloc[:,1:15]-min_FSR)/(max_FSR-min_FSR)
    out.iloc[:,15:17] = (X.iloc[:,15:17]-min_ACC)/(max_ACC-min_ACC)
    return out

def pre_process_data(X,order,cutoff,buffLen,numvars):
    X_acc = process_acc(X,order=order,cutoff=cutoff)
    norm_X = normalize_MLR(X_acc)
    X_segments, labels = segment_values(norm_X,buffLen,numvars) #[*,30,20]
    X_segments = X_segments.reshape(len(X_segments), 1, buffLen, numvars)
    X = get_max_features(X_segments) # See excel sheet 'Features Legend' for description of features
    Y = labels
#    labels = ['FSR1_R_mean','FSR2_R_mean','FSR3_R_mean','FSR4_R_mean','FSR5_R_mean','FSR6_R_mean','FSR7_R_mean',
#          'FSR1_L_mean','FSR2_L_mean','FSR3_L_mean','FSR4_L_mean','FSR5_L_mean','FSR6_L_mean','FSR7_L_mean',
#          'acc_R_mean','acc_L_mean',
#          'FSR1_R_STD','FSR2_R_STD','FSR3_R_STD','FSR4_R_STD','FSR5_R_STD','FSR6_R_STD','FSR7_R_STD',
#          'FSR1_L_STD','FSR2_L_STD','FSR3_L_STD','FSR4_L_STD','FSR5_L_STD','FSR6_L_STD','FSR7_L_STD',
#          'acc_R_STD','acc_L_STD',
#          'FSR1_R_med','FSR2_R_med','FSR3_R_med','FSR4_R_med','FSR5_R_med','FSR6_R_med','FSR7_R_med',
#          'FSR1_L_med','FSR2_L_med','FSR3_L_med','FSR4_L_med','FSR5_L_med','FSR6_L_med','FSR7_L_med',
#          'acc_R_med','acc_L_med',
#          'FSR1_R_CS','FSR2_R_CS','FSR3_R_CS','FSR4_R_CS','FSR5_R_CS','FSR6_R_CS','FSR7_R_CS',
#          'FSR1_L_CS','FSR2_L_CS','FSR3_L_CS','FSR4_L_CS','FSR5_L_CS','FSR6_L_CS','FSR7_L_CS',
#          'acc_R_CS','acc_L_CS',
#          'FSR1_R_MDD','FSR2_R_MDD','FSR3_R_MDD','FSR4_R_MDD','FSR5_R_MDD','FSR6_R_MDD','FSR7_R_MDD',
#          'FSR1_L_MDD','FSR2_L_MDD','FSR3_L_MDD','FSR4_L_MDD','FSR5_L_MDD','FSR6_L_MDD','FSR7_L_MDD',
#          'acc_R_MDD','acc_L_MDD',
#          'sum_FSR_means_R','sum_FSR_means_L','sum_FSR_means_LR']
    labels = ['FSR1_R_mean','FSR2_R_mean','FSR3_R_mean','FSR4_R_mean','FSR5_R_mean','FSR6_R_mean','FSR7_R_mean',
          'FSR1_L_mean','FSR2_L_mean','FSR3_L_mean','FSR4_L_mean','FSR5_L_mean','FSR6_L_mean','FSR7_L_mean',
          'acc_R_mean','acc_L_mean',
          'FSR1_R_STD','FSR2_R_STD','FSR3_R_STD','FSR4_R_STD','FSR5_R_STD','FSR6_R_STD','FSR7_R_STD',
          'FSR1_L_STD','FSR2_L_STD','FSR3_L_STD','FSR4_L_STD','FSR5_L_STD','FSR6_L_STD','FSR7_L_STD',
          'acc_R_STD','acc_L_STD']
    
    X = pd.DataFrame(X,columns = labels)
    return X, Y
    
    
def video_analysis(video_file, solution, num_samples, save_file):
    #  Load video file
    cap = cv2.VideoCapture(video_file)

    #  Video file properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frames = int((num_samples/(1000/22))*fps) # how long the video should be in # of frames

    #  Modify solution array to fit video length
    sol_index = np.asarray(np.where(solution[:-1] != solution[1:]))
    sol_index = np.reshape(sol_index,(-1,1))
    index = sol_index*(end_frames/num_samples)
    index = index.astype(dtype=int)
    #index = np.insert(index,0,0)

    sol_vid = np.empty(end_frames,dtype=int)
    for i in range (0,len(index)-1):
        if i==0:
            sol_vid[0:index[(1,0)]] = solution[sol_index[(i,0)]-1]
        sol_vid[index[(i,0)]:index[(i+1,0)]] = solution[sol_index[(i+1,0)]-2]
    sol_vid[index[(-1,0)]:]=solution[-1]

    #  Create text version of solution array to print on video
    txt_sol = []
    for i in range(0,len(sol_vid)):
        if sol_vid[i]==1:
            txt_sol.append('SITTING')
        if sol_vid[i]==2:
            txt_sol.append('STANDING')
        if sol_vid[i]==3:
            txt_sol.append('WALKING')

    #  Play video with solutions on the screen and save file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_file, fourcc, fps, (1080,608)) 
    count=0
    while True & (count<end_frames):
        ret, frame = cap.read()
        cv2.putText(img=frame,
                    text=txt_sol[count],
                    org = (int(40), int(frameHeight-40)), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale = 2, 
                    color = (0, 255, 0))
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(int(1000/fps)-5) & 0xFF == ord('q'): #this is where you can change the framerate
            break
        count = count + 1

    cap.release()
    cv2.destroyAllWindows() 

# Function to train SVM. Returns a trained model to be passed to test_SVM
def train_SVM(train_data,buffLen,numvars,C,select_feats=np.arange(106)):
    
    #normalize and buffer data
    train_data = normalize_3(train_data)
    train_segments, train_labels = segment_values(train_data,buffLen)
    train_x = train_segments.reshape(len(train_segments), 1, buffLen, numvars)
    
    #train SVM
    X = get_max_features(train_x)
    X = np.take(X,select_feats,axis=1)
    Y = train_labels

    model = svm.SVC(C=C, kernel='sigmoid', gamma='auto')
    model.fit(X,Y)
    
    #old version
#    #buffer data
#    train_segments, train_labels = segment_values_NO(train_data,buffLen)
#    train_x = train_segments.reshape(len(train_segments), 1, buffLen, numvars)
#    
#    #train SVM
#    X = get_features(normalize_2(train_x))
#    X = np.reshape(X,[-1,X.shape[1]*X.shape[2]])
#    Y = train_labels
#
#    model = svm.SVC(C=4, kernel='sigmoid', gamma='auto', max_iter=10000)
#    model.fit(X,Y)
    
    return model

# Function to train SVM. Returns a trained model to be passed to test_SVM
def train_SVM_PCA(train_data,buffLen,numvars,select_feats=np.arange(106)):
    
    #normalize and buffer data
    train_data = normalize_3(train_data)
    train_segments, train_labels = segment_values(train_data,buffLen)
    train_x = train_segments.reshape(len(train_segments), 1, buffLen, numvars)
    
    #train SVM
    X = get_max_features(train_x)
    X = np.take(X,select_feats,axis=1)
    Y = train_labels
    
    pca = PCA(0.98)
    pca.fit(X)
    X_transformed = pca.transform(X)

    model = svm.SVC(C=4, kernel='sigmoid', gamma='auto', max_iter=10000)
    model.fit(X_transformed,Y)
    
    return model,X_transformed,pca

# Function to get predicted solutions from trained model. 
# Returns solutions (y_) and actual solutions (test_labels) 
def test_SVM(test_data,buffLen,numvars,model,select_feats=np.arange(106)):
    
    #Normalize and buffer data
    test_data = normalize_3(test_data)
    test_segments, test_labels = segment_values(test_data,buffLen)
    test_x = test_segments.reshape(len(test_segments), 1, buffLen, numvars)
    
    #get solutions
    T = get_max_features(test_x)
    T = np.take(T,select_feats,axis=1)
    y_ = model.predict(T)
    
    #0ld version
#    test_segments, test_labels = segment_values_NO(test_data,buffLen)
#    test_x = test_segments.reshape(len(test_segments), 1, buffLen, numvars)
#    
#    #get solutions
#    T = get_features(normalize_2(test_x))
#    T = np.reshape(T,[-1,T.shape[1]*T.shape[2]])
#    y_ = model.predict(T)
    
    return y_, test_labels

# Function to get predicted solutions from trained model. 
# Returns solutions (y_) and actual solutions (test_labels) 
def test_SVM_PCA(pca,test_data,buffLen,numvars,model,select_feats=np.arange(106)):
    
    #Normalize and buffer data
    test_data = normalize_3(test_data)
    test_segments, test_labels = segment_values(test_data,buffLen)
    test_x = test_segments.reshape(len(test_segments), 1, buffLen, numvars)
    
    #get solutions
    T = get_max_features(test_x)
    T = np.take(T,select_feats,axis=1)
    T_transformed = pca.transform(T)
    y_ = model.predict(T_transformed)
    
    return y_, test_labels

# Plots results along with FSR data and actual activity state if included in test_data
def plot_results(test_data,y_out_val):
    fig,ax=plt.subplots(sharex=True, nrows=3, ncols=1)
    ax[0].plot(test_data['FSR1_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR2_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR3_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR4_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR5_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR6_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR7_R'], linewidth=0.5)
    ax[0].set_ylim((0,900))
    ax[0].legend()
    ax[0].set_title('FSR data Right')
    ax[0].set_ylabel('FSR Output')
    
    ax[1].plot(test_data['FSR1_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR2_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR3_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR4_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR5_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR6_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR7_L'], linewidth=0.5)
    ax[1].set_ylim((0,900))
    ax[1].legend()
    ax[1].set_title('FSR Data Left')
    ax[1].set_ylabel('FSR Output')
    
    ax[2].plot(test_data['ActivityState'], label='Actual')
    ax[2].plot(y_out_val, label='Predicted')
#    sampfreq = np.arange(0,len(test_data),90)
#    ax[2].plot(sampfreq, np.ones(len(sampfreq))*2, 'bo')
    ax[2].set_ylim((0.5,3.5))
    ax[2].legend()
    ax[2].set_title('Activity State.')
    ax[2].set_xlabel('Sample Number')
    ax[2].set_ylabel('1=Sit, 2=Stand, 3=Walk')
    
    mng=plt.get_current_fig_manager() 
    mng.window.showMaximized() #maximize figure 
    plt.show()
    
    return

# Plots L and R FSR data and overlays synchronization metrics
def check_synchronization(test_data, threshold, min_freq, max_freq):
    dist = len(test_data)
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
    
    #distance between threshold crossings
    diffs_R = np.asarray([y - x for x,y in zip(crosslocs_R,crosslocs_R[1:])])
    j = 0
    sol_R = []
    for i in range(1,len(diffs_R)):
        curval = diffs_R[i]
        if curval>min_freq and curval<max_freq:
            j = j+1
            if j>2:
                sol_R = np.append(sol_R,crosslocs_R[i-2])
        else:
            j=0
    sol_R_mid = (sol_R[1:] + sol_R[:-1]) / 2
    
    diffs_L = np.asarray([y - x for x,y in zip(crosslocs_L,crosslocs_L[1:])])
    j = 0
    sol_L = []
    for i in range(1,len(diffs_L)):
        curval = diffs_L[i]
        if curval>min_freq and curval<max_freq:
            j = j+1
            if j>2:
                sol_L = np.append(sol_L,crosslocs_L[i-2])
        else:
            j=0
    sol_L_mid = (sol_L[1:] + sol_L[:-1]) / 2
    
    fig,ax=plt.subplots(sharex=True, nrows=2, ncols=1)
    ax[0].plot(test_data['FSR1_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR2_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR3_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR4_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR5_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR6_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR7_R'], linewidth=0.5)
    ax[0].plot(sol_R,np.ones(len(sol_R))*threshold,'bo')
    for xc in sol_R:
        ax[0].axvline(x=xc, color='b')
    for xc in sol_L_mid:
        ax[0].axvline(x=xc, color='r')
    ax[0].set_ylim((0,900))
    ax[0].legend()
    ax[0].set_title('FSR data Right')
    ax[0].set_ylabel('FSR Output')
    
    ax[1].plot(test_data['FSR1_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR2_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR3_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR4_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR5_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR6_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR7_L'], linewidth=0.5)
    ax[1].plot(sol_L,np.ones(len(sol_L))*threshold,'ro')
    for xc in sol_L:
        ax[1].axvline(x=xc, color='r')
    for xc in sol_R_mid:
        ax[1].axvline(x=xc, color='b')
    ax[1].set_ylim((0,900))
    ax[1].legend()
    ax[1].set_title('FSR Data Left')
    ax[1].set_ylabel('FSR Output')
    
    mng=plt.get_current_fig_manager() 
    mng.window.showMaximized() #maximize figure 
    plt.show()
    
    return

# Plots results along with FSR data and actual activity state if included in test_data
def save_plot_results(test_data,y_out_val,filename,figsize):
    fig,ax=plt.subplots(sharex=True, nrows=3, ncols=1,figsize=figsize)
    ax[0].plot(test_data['FSR1_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR2_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR3_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR4_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR5_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR6_R'], linewidth=0.5)
    ax[0].plot(test_data['FSR7_R'], linewidth=0.5)
    ax[0].set_ylim((0,900))
    ax[0].legend()
    ax[0].set_title('FSR data Right')
    ax[0].set_ylabel('FSR Output')
    
    ax[1].plot(test_data['FSR1_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR2_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR3_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR4_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR5_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR6_L'], linewidth=0.5)
    ax[1].plot(test_data['FSR7_L'], linewidth=0.5)
    ax[1].set_ylim((0,900))
    ax[1].legend()
    ax[1].set_title('FSR Data Left')
    ax[1].set_ylabel('FSR Output')
    
    ax[2].plot(test_data['ActivityState'], label='Actual')
    ax[2].plot(y_out_val, label='Predicted')
    ax[2].set_ylim((0.5,3.5))
    ax[2].legend()
    ax[2].set_title('Activity State.')
    ax[2].set_xlabel('Sample Number')
    ax[2].set_ylabel('1=Sit, 2=Stand, 3=Walk')
    
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close('all')
    
    return

# reshapes results to match size of test data
def reshape_ybar(y_,test_data):
    if len(np.unique(y_)) < 2:
        print("one soution only")
        y_out_val = np.ones(len(test_data),dtype=int)*y_[0]
        
    if len(np.unique(y_)) >= 2:
        y_index = np.asarray(np.where(y_[:-1] != y_[1:]))
        y_index = np.reshape(y_index,(-1,1))
        index = y_index*(len(test_data)/len(y_))
        index = index.astype(dtype=int)
        y_out_val = np.empty(len(test_data),dtype=int)
        for i in range (0,len(index)):
            if i==0:
                y_out_val[0:index[(0,0)]] = y_[y_index[(i,0)]]
            else:
                y_out_val[index[(i-1,0)]:index[(i,0)]] = y_[y_index[(i,0)]]
        y_out_val[(index[(-1,0)]):]=y_[-1]
        #this section can only be used with a buffLen of 40 when subject is sitting to start
        y_out_val = np.insert(y_out_val,0,(np.asarray(np.ones(40))))
        y_out_val = y_out_val[0:-40]
        
    return y_out_val

# Calculates CM and accuracy
def get_cm(test_labels, y_,title):
    cm = confusion_matrix(test_labels, y_)
    
    # code to plot CM
#    plt.figure(2)
#    sn.heatmap(cm, annot=True, fmt='g',annot_kws={"size": 30}, 
#               square=True, xticklabels=['Sit','Stand','Walk'],
#               yticklabels=['Sit','Stand','Walk'])
#    plt.title(title)
    
    stat = np.zeros([3,4])
    for i in range(0,3):
        stat[i,0] = cm[i,i]
        stat[i,1] = (cm[0,0]+cm[1,1]+cm[2,2])-cm[i,i]
        stat[i,2] = np.sum(cm[:,i])-cm[i,i]
        stat[i,3] = np.sum(cm[i,:])-cm[i,i]
    
    stat = np.mean(stat,axis=0)
    TP = stat[0]
    TN = stat[1]
    FP = stat[2]
    FN = stat[3]
    
    #accuracy = (TP+TN)/(TP+TN+FP+FN) #(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
    accuracy = (np.trace(cm))/sum(sum(cm))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F = 2*(precision*recall)/(precision+recall) #something is fucked up about this
    
    #print('Accuracy: ', (accuracy*100), 'F measure: ', (F*100))
    #print('Accuracy: ', (accuracy*100), '(',(np.trace(cm)),'/',sum(sum(cm)),')')
    #print(cm)
    return cm, accuracy

def get_training_data(comp):
    #import all data
    # NOTE: SID21,22,23 and 27 do not have calibration data
    # NOTE: SID18 should be left out as it's data causes issues for the algorithm
    
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


    # Assembly setup
    #modify this list to determine which Subjects to include in the training / testing data.
    # This vvv is a complete list, copy this then modify as needed.
    #SID_i = ['SID01','SID02','SID03','SID04','SID05','SID06','SID07','SID08','SID09','SID10',
    #         'SID11','SID12','SID13','SID14','SID15','SID16','SID17','SID18','SID19','SID20',
    #         'SID24','SID25','SID26','SID28','SID29','SID30','SID31','SID32','SID33','SID34','SID35','SID36']
    
    # selected group of training data based on functioning sensors
    SID_i = ['SID05','SID06','SID07','SID08','SID09','SID10',
             'SID12','SID13','SID14','SID15','SID16','SID17','SID19','SID20',
             'SID24','SID25','SID26','SID30','SID32','SID34','SID36']
    
    
    #Dictionary pointing to all the dataframes imported above to be used for iterating through.
    SID_dict = {'SID01':SID01,'SID02':SID02,'SID03':SID03,'SID04':SID04,'SID05':SID05,'SID06':SID06,'SID07':SID07,'SID08':SID08,'SID09':SID09,'SID10':SID10,
                'SID11':SID11,'SID12':SID12,'SID13':SID13,'SID14':SID14,'SID15':SID15,'SID16':SID16,'SID17':SID17,'SID18':SID18,'SID19':SID19,'SID20':SID20,
                'SID24':SID24,'SID25':SID25,'SID26':SID26,'SID28':SID28,'SID29':SID29,'SID30':SID30,'SID31':SID31,'SID32':SID32,'SID33':SID33,'SID34':SID34,'SID35':SID35,'SID36':SID36}
    
    #Format for calling a specific subject's data
    #test_data = SID_dict[SID_i[i]]

    train_data = SID_dict[SID_i[0]]
    for i in range(0,len(SID_i)):
        df = SID_dict[SID_i[i]]
        train_data = train_data.append(df).reset_index(drop=True)
        
    return train_data

def combined_data_two_insole(data_right, data_left, ts_right, ts_left):
    column_names = ['time', 'FSR1_R', 'FSR2_R', 'FSR3_R', 'FSR4_R', 
                    'FSR5_R', 'FSR6_R', 'FSR7_R', 'X_R', 'Y_R', 'Z_R', 'ActivityState_R',
                    'time_L', 'FSR1_L', 'FSR2_L', 'FSR3_L', 'FSR4_L', 
                    'FSR5_L', 'FSR6_L', 'FSR7_L', 'X_L', 'Y_L', 'Z_L', 'ActivityState']
    
    # Set offset times based on video data
    ts_data_right = ts_right #seconds elapsed at start of data recording
    ts_data_left = ts_left #seconds elapsed at start of data recording
    
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
    
    return SynchronizedData

def offset_data(data,shift):
    
    #positive shift offsets R foot data so it starts after L foot, negative value offsets L foot data
    data_offset = pd.DataFrame(data)
    data_offset['FSR1_R'] = data_offset['FSR1_R'].shift(shift)
    data_offset['FSR2_R'] = data_offset['FSR2_R'].shift(shift)
    data_offset['FSR3_R'] = data_offset['FSR3_R'].shift(shift)
    data_offset['FSR4_R'] = data_offset['FSR4_R'].shift(shift)
    data_offset['FSR5_R'] = data_offset['FSR5_R'].shift(shift)
    data_offset['FSR6_R'] = data_offset['FSR6_R'].shift(shift)
    data_offset['FSR7_R'] = data_offset['FSR7_R'].shift(shift)
    data_offset['X_R'] = data_offset['X_R'].shift(shift)
    data_offset['Y_R'] = data_offset['Y_R'].shift(shift)
    data_offset['Z_R'] = data_offset['Z_R'].shift(shift)
    data_offset = data_offset.dropna().reset_index(drop=True)
    
    return data_offset

def downsample(data,dec):
    data_down = data.iloc[::dec, :].reset_index(drop=True)
    
    return data_down