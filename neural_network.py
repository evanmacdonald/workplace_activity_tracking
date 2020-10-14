# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:04:49 2019

@author: evanm
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from sklearn.metrics import confusion_matrix
import Kintec_Functions as kf

#%% import all data
# NOTE: SID21,22,23 do not have calibration data
# NOTE: SID18 and SID27 should be left out as it's data causes issues for the algorithm
#print('Importing Data...')
#
#comp = 'evanm'
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID01/SynchronizedData/SID01_Calibration.csv')
#SID01 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID02/SynchronizedData/SID02_Calibration.csv')
#SID02 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID03/SynchronizedData/SID03_Calibration.csv')
#SID03 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID04/SynchronizedData/SID04_Calibration.csv')
#SID04 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID05/SynchronizedData/SID05_Calibration.csv')
#SID05 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID06/SynchronizedData/SID06_Calibration.csv')
#SID06 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID07/SynchronizedData/SID07_Calibration.csv')
#SID07 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID08/SynchronizedData/SID08_Calibration.csv')
#SID08 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID09/SynchronizedData/SID09_Calibration.csv')
#SID09 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID10/SynchronizedData/SID10_Calibration.csv')
#SID10 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID11/SynchronizedData/SID11_Calibration.csv')
#SID11 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID12/SynchronizedData/SID12_Calibration.csv')
#SID12 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID13/SynchronizedData/SID13_Calibration.csv')
#SID13 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID14/SynchronizedData/SID14_Calibration.csv')
#SID14 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID15/SynchronizedData/SID15_Calibration.csv')
#SID15 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID16/SynchronizedData/SID16_Calibration.csv')
#SID16 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID17/SynchronizedData/SID17_Calibration.csv')
#SID17 = pd.read_csv(file)
#
##leave this one out, it is not good data
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID18/SynchronizedData/SID18_Calibration.csv')
#SID18 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID19/SynchronizedData/SID19_Calibration.csv')
#SID19 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID20/SynchronizedData/SID20_Calibration.csv')
#SID20 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID24/SynchronizedData/SID24_Calibration.csv')
#SID24 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID25/SynchronizedData/SID25_Calibration.csv')
#SID25 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID26/SynchronizedData/SID26_Calibration.csv')
#SID26 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID28/SynchronizedData/SID28_Calibration.csv')
#SID28 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID29/SynchronizedData/SID29_Calibration.csv')
#SID29 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID30/SynchronizedData/SID30_Calibration.csv')
#SID30 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID31/SynchronizedData/SID31_Calibration.csv')
#SID31 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID32/SynchronizedData/SID32_Calibration.csv')
#SID32 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID33/SynchronizedData/SID33_Calibration.csv')
#SID33 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID34/SynchronizedData/SID34_Calibration.csv')
#SID34 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID35/SynchronizedData/SID35_Calibration.csv')
#SID35 = pd.read_csv(file)
#
#file = open('C:/Users/'+comp+'/sfuvault/Thesis/Trial Data/Full Trial/SID36/SynchronizedData/SID36_Calibration.csv')
#SID36 = pd.read_csv(file)
#
#print('Data imported...')

#%% Leave one out setup
#
##modify this list to determine which Subjects to include in the training / testing data.
## This vvv is a complete list, copy this then modify as needed.
##SID_i = ['SID01','SID02','SID03','SID04','SID05','SID06','SID07','SID08','SID09','SID10',
##         'SID11','SID12','SID13','SID14','SID15','SID16','SID17','SID18','SID19','SID20',
##         'SID24','SID25','SID26','SID28','SID29','SID30','SID31','SID32','SID33','SID34','SID35','SID36']
#
#SID_i = ['SID05','SID06','SID07','SID08','SID09','SID10',
#         'SID12','SID13','SID14','SID15','SID16','SID17','SID19','SID20',
#         'SID24','SID25','SID26','SID30','SID32','SID34','SID36']
#
##Dictionary pointing to all the dataframes imported above to be used for iterating through.
#SID_dict = {'SID01':SID01,'SID02':SID02,'SID03':SID03,'SID04':SID04,'SID05':SID05,'SID06':SID06,'SID07':SID07,'SID08':SID08,'SID09':SID09,'SID10':SID10,
#            'SID11':SID11,'SID12':SID12,'SID13':SID13,'SID14':SID14,'SID15':SID15,'SID16':SID16,'SID17':SID17,'SID18':SID18,'SID19':SID19,'SID20':SID20,
#            'SID24':SID24,'SID25':SID25,'SID26':SID26,'SID28':SID28,'SID29':SID29,'SID30':SID30,'SID31':SID31,'SID32':SID32,'SID33':SID33,'SID34':SID34,'SID35':SID35,'SID36':SID36}
#
##Format for calling a specific subject's data
##test_data = SID_dict[SID_i[i]]

#%% BUFFER
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
#%% Neural Network Setup

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_fc_layer(x,
                 num_inputs,
                 num_outputs):
    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    #create activations with biases added in and ReLU
    activations = tf.nn.relu(tf.add(tf.matmul(x, weights), biases))
    return activations

def new_logits(x,
               num_inputs,
               num_outputs):
    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    #create activations with biases added in and ReLU
    activations = tf.matmul(x, weights) + biases
    return activations

def network_setup(numFeats,buffLen,num_nodes):
    #Number of ouput classes
    num_classes = 3

    #Other parameters
    learning_rate = 0.0001
    
    '''Placeholder variables'''
    #Input 'images'
    X = tf.placeholder(tf.float32, shape=[None, numFeats], name='X')
    #output[#,20,90,1][numsamples, numVars, buffLen, 1]
    Y = tf.placeholder(tf.float32, shape=[None,num_classes], name='Y')
    #output[#,3][numsamples,num_classes]
    Y_cls = tf.argmax(Y, axis=1)
    
    '''Create fully-connected layer'''
    f1 = new_fc_layer(X, numFeats, num_nodes)
    
    '''Create fully-connected output layer'''
    f2 = new_logits(f1, num_nodes, num_classes)
    
    '''Create softmax output'''
    y_ = tf.nn.softmax(f2)
    #Produces index of largest element
    y_pred_cls = tf.argmax(y_, axis=1)
    
    '''Cost function and optimizer'''
    # This is where the most improvement could come from
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=f2, labels=Y)
    cost = tf.reduce_mean(cross_entropy) #this is the mean of the cross entropy of all the image classifications
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    '''Performance Measures'''
    correct_prediction = tf.equal(y_pred_cls, Y_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #shockingly this actually works as an accuracy measure when you are using 1 and 0
    return X, Y, Y_cls, y_, y_pred_cls, cross_entropy, cost, optimizer, correct_prediction, accuracy 

def next_batch(num, x, y):
    #Return a total of 'num' random samples and labels. 
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [x[i] for i in idx]
    labels_shuffle = [y[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def optimize(num_iterations,batch_size,x,y,total_iterations,session,X,Y,optimizer,cost,accuracy):
    # Start-time used for printing time-usage below.
#    start_time = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch = next_batch(batch_size, x, y)
        # Put the batch into a dict
        feed_dict_train = {X: x_batch, Y: y_true_batch}
        # Run the optimizer using this batch of training data.
        session.run([optimizer, cost], feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            acc,cst = session.run([accuracy,cost], feed_dict=feed_dict_train)
#            msg = "Iteration: {0:>6}, Tr. Accuracy: {1:>6.1%}, Cost: {2:>6.2}"
#            print(msg.format(i + 1, acc, cst))
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    # Get final stats
    acc,cst = session.run([accuracy,cost], feed_dict=feed_dict_train)
    # Ending time.
#    end_time = time.time()
#    time_dif = end_time - start_time
#    print("Training time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return acc, cst, total_iterations

def plot_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=test_y_cls, y_pred=cls_pred)
    print(cm)
    sn.heatmap(cm, annot=True, fmt='g',annot_kws={"size": 30}, 
               square=True, xticklabels=['Sit','Stand','Walk'],
               yticklabels=['Sit','Stand','Walk'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return
    

def check_accuracy(x,y,y_cls,session,X,Y,y_pred_cls):
    #Initialize the array to be filled in as it goes
    num_test = len(x)


    #Method for getting predicted classifications using a batch size
#    cls_pred = np.zeros(shape=num_test, dtype=np.int)
#    test_batch_size = 50
#    i = 0

#    while i < num_test:
#        # The ending index for the next batch is denoted j.
#        j = min(i + test_batch_size, num_test)
#        # Get the images and labels from the test-set between index i and j.
#        images = x[i:j, :]
#        labels = y[i:j, :]
#        # Create a feed-dict with these images and labels.
#        feed_dict = {X: images, Y: labels}
#        # Calculate the predicted class using TensorFlow.
#        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
#        # Set the start-index for the next batch to the end-index of the current batch.
#        i = j
    feed_dict = {X: x, Y: y}
    
    # get predictions for test set
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

    # Create a boolean array whether each image is correctly classified.
    correct = (y_cls == cls_pred)
    # Calculate the number of correctly classified images.
    correct_sum = correct.sum()
    # Classification accuracy
    acc = float(correct_sum) / num_test
    # Print the accuracy.
#    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#    print(msg.format(acc, correct_sum, num_test))
    
#    msg = "Classification time usage: {0:>6.5} seconds"
#    print(msg.format(time_dif))

    # Plot some examples of mis-classifications, if desired.
#    if show_example_errors:
#        print("Example errors:")
#        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    return acc, correct_sum, num_test, cls_pred


#%% Optimize loop

#print('Getting solutions...')
#
#num_nodes = 500
#
##setup network
#X, Y, Y_cls, y_, y_pred_cls, cross_entropy, cost, optimizer, correct_prediction, accuracy = network_setup(len(select_feats),buffLen,num_nodes)
## TensorFlow portion
#'''TensorFlow Session'''
#session = tf.Session()
#session.run(tf.global_variables_initializer())
#
#'''Optimization'''
#train_batch_size = 1000
## Counter for total number of iterations performed so far.
#total_iterations = 0
## Optimize network
#cost_step = 10 #arbitrarily large number for cost
#while cost_step > 0.01 and total_iterations < 200000:
#    acc_train, cost_step = optimize(1,train_batch_size,train_x,train_y)
#
##acc_train, cost_step = optimize(80)
#        
##Final stats    
#msg = "Iterations: {0:>2}, Tr. Accuracy: {1:>3.1%}, Cost: {2:>2.2}"
#print(msg.format(total_iterations, acc_train, cost_step))
#
##check test accuracy
#acc_test, test_correct, test_total = check_accuracy(test_x,test_y,test_y_cls,cm=True)
#
#session.close()

#%%
def train_neural_net(train_x,train_y,test_x,test_y,test_y_cls,select_feats,num_nodes,buffLen):

    #setup network
    X, Y, Y_cls, y_, y_pred_cls, cross_entropy, cost, optimizer, correct_prediction, accuracy = network_setup(len(select_feats),buffLen,num_nodes)
    # TensorFlow portion
    '''TensorFlow Session'''
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    '''Optimization'''
    train_batch_size = 1000
    # Counter for total number of iterations performed so far.
    total_iterations = 0
    # Optimize network
    cost_step = 10 #arbitrarily large number for cost
    while cost_step > 0.01 and total_iterations < 15000:
        step = 1
        acc_train, cost_step, total_iterations = optimize(step,train_batch_size,train_x,train_y,total_iterations,
                                                          session,X,Y,optimizer,cost,accuracy)
    
    #acc_train, cost_step = optimize(80)
            
    #Final stats    
#    msg = "Iterations: {0:>2}, Tr. Accuracy: {1:>3.1%}, Cost: {2:>2.2}"
#    print(msg.format(total_iterations, acc_train, cost_step))
    
    #check test accuracy
    acc_test, test_correct, test_total, y_ = check_accuracy(test_x,test_y,test_y_cls,
                                                        session,X,Y,y_pred_cls)
    
    session.close()
    
    return acc_train, acc_test, cost_step, total_iterations, y_

