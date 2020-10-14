# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:22:13 2018

@author: Evan Macdonald

Code to train and test a Convolutional Neural Network to recognize activities
Designed for CMPT884 class and for later use in masters thesis

HOW TO RUN
 start with fresh run of script (clear all variables and re-start kernel)
 optimize(num_iterations=#) - Can start small, check accuracy and then keep going from there
 check_accuracy(show_cm=True) - shows accuracy and CM of current state of network
 plot_solutions() - shows results on a plot
"""

#Imports
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
from data_assembly_two_insole_CMPT884 import combined_data_two_insole
from Kintec_Functions_CMPT884 import segment_values_NO
from Kintec_Functions_CMPT884 import normalize_2


#%% import all data using data assembly script

print('Importing Data...')
train_data, test_data = combined_data_two_insole()
print('Data imported...')

#%% BUFFER

print('Buffering Data...')
# buffer data for input into classifier
buffLen =  90 #45, 90, 135, 180, 225
numVars = 20
overlap = buffLen/4 # Number of datapoints to overlap in the buffers

train_segments, train_labels = segment_values_NO(train_data,buffLen)
train_y = np.asarray(pd.get_dummies(train_labels), dtype = np.int8)
train_x = normalize_2(train_segments.reshape(len(train_segments), 1, buffLen, numVars))
train_x = np.rot90(train_x, k=3, axes=(2,3))
train_x = np.transpose(train_x,(0,2,3,1))

test_segments, test_labels = segment_values_NO(test_data,buffLen)
test_y = np.asarray(pd.get_dummies(test_labels), dtype = np.int8)
test_y_cls = np.argmax(test_y, axis=1)
test_x = normalize_2(test_segments.reshape(len(test_segments), 1, buffLen, numVars))
test_x = np.rot90(test_x, k=3, axes=(2,3))
test_x = np.transpose(test_x,(0,2,3,1))
print('Data buffered...')

#%% Heatmap of input image
#row_names = ['FSR1R', 'FSR2R', 'FSR3R', 'FSR4R', 'FSR5R', 'FSR6R', 'FSR7R',
#                    'FSR1L', 'FSR2L', 'FSR3L', 'FSR4L', 'FSR5L', 'FSR6L', 'FSR7L', 
#                    'XR', 'YR', 'ZR', 'XL', 'YL', 'ZL']
#ax = sn.heatmap(train_x[105,:,:,0], #@ buffLen 90 - sit = 20, stand = 80, walk = 105
#                cmap="gray", 
#                square=True,
#                cbar = False, 
#                yticklabels=row_names, 
#                xticklabels=False)
#plt.yticks(rotation=0) 

#%% CNN


#Helper functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Layer constructors
def new_conv_layer(x,
                   num_input_channels,
                   filter_width,
                   filter_height,
                   num_filters,
                   pool_height,
                   pool_width):
    #Shape of the convolution kernel
    # [buffLen, 20, 1, num_filters]
    shape = [filter_width, filter_height, num_input_channels, num_filters]
    #Create weights
    weights = new_weights(shape=shape)
    #Create biases
    biases = new_biases(length=num_filters)
    #Create convolution layer activations
    activations = tf.nn.conv2d(input=x, filter=weights, strides=[1,1,1,1], padding='SAME')
    #Add in biases
    activations = tf.add(activations, biases)
    #Complete max pooiling
    activations = tf.nn.max_pool(value=activations, 
                                 ksize=[1,pool_height,pool_width,1],
                                 strides=[1,pool_height,pool_width,1],
                                 padding='SAME')
    #Rectified Linear Unit (ReLU)
    # calcs max(x,0) for each input pixel
    # note this is okay to do after max pooling since relu(max_pool(x)) == max_pool(relu(x)) 
    activations = tf.nn.relu(activations)
    #weights are being returned since there may be a use to  look at them later
    return activations, weights

def flatten_layer(layer):
    #get shape of input layer
    # [#, img_height, img_width, 1]
    layer_shape = layer.get_shape()
    #Calculate number of features 
    # =img_height*img_width*1
    num_features = layer_shape[1:4].num_elements()
    #Reshape the layer to flat
    # [#, img_height*img_width*1]
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

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



def network_setup(filter_width,
                  num_filters,
                  pool_height,
                  pool_width,
                  num_nodes,):
    #Number of ouput classes
    num_classes = 3
    filter_height = 20 #numVars

    #Other parameters
    learning_rate = 0.0001
    '''Placeholder variables'''
    #Input 'images'
    X = tf.placeholder(tf.float32, shape=[None, numVars, buffLen, 1], name='X')
    #output[#,20,90,1][numsamples, numVars, buffLen, 1]
    Y = tf.placeholder(tf.float32, shape=[None,num_classes], name='Y')
    #output[#,3][numsamples,num_classes]
    Y_cls = tf.argmax(Y, axis=1)

    '''Create convolution layer'''
    c1, w1 = new_conv_layer(X, 1, filter_width, filter_height, num_filters, pool_height, pool_width)
    
    '''Create flattened layer'''
    flat, num_features = flatten_layer(c1)
    
    '''Create fully-connected layer'''
    f1 = new_fc_layer(flat, num_features, num_nodes)
    
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

def optimize(num_iterations):
    global total_iterations
    # Start-time used for printing time-usage below.
#    start_time = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch = next_batch(train_batch_size, train_x, train_y)
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
    return acc, cst

# Maybe develop this later
#def plot_example_errors(cls_pred, correct):
#    # cls_pred is an array of the predicted class-number for
#    # all images in the test-set.
#
#    # correct is a boolean array whether the predicted class
#    # is equal to the true class for each image in the test-set.
#
#    # Negate the boolean array.
#    incorrect = (correct == False)
#    
#    # Get the images from the test-set that have been
#    # incorrectly classified.
#    images = test_x[incorrect]
#    
#    # Get the predicted classes for those images.
#    cls_pred = cls_pred[incorrect]
#
#    # Get the true classes for those images.
#    cls_true = test_y[incorrect]
#    
#    # Plot the first 9 images. (this is something to do in the future)
##    plot_images(images=images[0:9],
##                cls_true=cls_true[0:9],
##                cls_pred=cls_pred[0:9])
#    return images, cls_pred, cls_true

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


def check_accuracy(cm=False):
    #Initialize the array to be filled in as it goes
    num_test = len(test_x)


    #Method for getting predicted classifications using a batch size
#    cls_pred = np.zeros(shape=num_test, dtype=np.int)
#    test_batch_size = 50
#    i = 0

#    while i < num_test:
#        # The ending index for the next batch is denoted j.
#        j = min(i + test_batch_size, num_test)
#        # Get the images and labels from the test-set between index i and j.
#        images = test_x[i:j, :]
#        labels = test_y[i:j, :]
#        # Create a feed-dict with these images and labels.
#        feed_dict = {X: images, Y: labels}
#        # Calculate the predicted class using TensorFlow.
#        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
#        # Set the start-index for the next batch to the end-index of the current batch.
#        i = j
    images = test_x
    labels = test_y
    feed_dict = {X: images, Y: labels}
    
    #start timer
    start_time = time.time()
    
    # get predictions for test set
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    
    #end timer
    end_time = time.time()
    time_dif = end_time - start_time

    # Create a boolean array whether each image is correctly classified.
    correct = (test_y_cls == cls_pred)
    # Calculate the number of correctly classified images.
    correct_sum = correct.sum()
    # Classification accuracy
    acc = float(correct_sum) / num_test
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    
#    msg = "Classification time usage: {0:>6.5} seconds"
#    print(msg.format(time_dif))

    # Plot some examples of mis-classifications, if desired.
#    if show_example_errors:
#        print("Example errors:")
#        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if cm:
        plot_confusion_matrix(cls_pred=cls_pred)
    return acc, correct_sum, num_test

def plot_solutions():

    images = test_x
    labels = test_y
    feed_dict = {X: images, Y: labels}
    
    # get predictions for test set
    y_ = session.run(y_pred_cls, feed_dict=feed_dict)
    
    # Reshape y_ to match test_data
    y_index = np.asarray(np.where(y_[:-1] != y_[1:]))
    y_index = np.reshape(y_index,(-1,1))
    index = y_index*(len(test_data)/len(y_))
    index = index.astype(dtype=int)
    y_out_val = np.empty(len(test_data),dtype=int)
    for i in range (0,len(index)-1):
        if i==0:
            y_out_val[0:index[(1,0)]] = y_[y_index[(i,0)]-1]
        y_out_val[index[(i,0)]:index[(i+1,0)]] = y_[y_index[(i+1,0)]-2]
    y_out_val[index[(-1,0)]:]=y_[-1]
    y_out_val = y_out_val + 1
    
    # Reshape test_labels to match y_out_val
    yt_index = np.asarray(np.where(test_labels[:-1] != test_labels[1:]))
    yt_index = np.reshape(yt_index,(-1,1))
    index = yt_index*(len(test_data)/len(test_labels))
    index = index.astype(dtype=int)
    y_test_val = np.empty(len(test_data),dtype=int)
    for i in range (0,len(index)-1):
        if i==0:
            y_test_val[0:index[(1,0)]] = test_labels[yt_index[(i,0)]-1]
        y_test_val[index[(i,0)]:index[(i+1,0)]] = test_labels[yt_index[(i+1,0)]-2]
    y_test_val[index[(-1,0)]:]=test_labels[-1]
    
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

    ax[2].plot(y_test_val, label='Actual')
    ax[2].plot(y_out_val, label='Predicted')
    ax[2].set_ylim((0.5,3.5))
    ax[2].legend()
    ax[2].set_title('Activity State.')
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('1=Sit, 2=Stand, 3=Walk')

    mng=plt.get_current_fig_manager() 
    mng.window.showMaximized() #maximize figure 
    plt.show()

#%% Optimize loop
#Configuration of parameters (these are what you can modify)
filter_width = [14] 
num_filters = [45] #over 20 seems to not give much improvement
#Pooling layer parameters
pool_height = [3]
pool_width = [3]
#Fully-connected layer parameters
num_nodes = [10,20,40,60,80,100,200,300,600,800] #
combos = np.array(np.meshgrid(filter_width,
                              num_filters,
                              pool_height,
                              pool_width,
                              num_nodes)).T.reshape(-1,5)

#initialize blank solution arrays to store results
accList = np.zeros(len(combos))
correctList = np.zeros([len(combos),2])
iterList = np.zeros(len(combos))
costList = np.zeros(len(combos))

#iterate through all possible solutions
for i in range(len(combos)):
    #setup network
    X, Y, Y_cls, y_, y_pred_cls, cross_entropy, cost, optimizer, correct_prediction, accuracy = network_setup(combos[i,0], #filter_width
                                                                                                              combos[i,1], #num_filters
                                                                                                              combos[i,2], #pool_height
                                                                                                              combos[i,3], #pool_width
                                                                                                              combos[i,4],) #num_nodes
    # TensorFlow portion
    '''TensorFlow Session'''
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    '''Optimization'''
    train_batch_size = 1000
    train_total_batches = train_x.shape[0] // train_batch_size
    # Counter for total number of iterations performed so far.
    total_iterations = 0
    # Optimize network
#    cost_step = 10 #arbitrarily large number for cost
#    while cost_step > 0.01 and total_iterations < 2000:
#        acc_train, cost_step = optimize(1)
    
    acc_train, cost_step = optimize(80)
            
    #Final stats    
    msg = "Combo #: {0:>2}, Iterations: {1:>2}, Tr. Accuracy: {2:>3.1%}, Cost: {3:>2.2}"
    print(msg.format((i+1),total_iterations, acc_train, cost_step))
    #get accuracy stats
    accList[i], correctList[i,0], correctList[i,1] = check_accuracy()
    iterList[i] = total_iterations
    costList[i] = cost_step
