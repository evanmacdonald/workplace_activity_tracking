# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 08:34:24 2019

@author: evanm
"""
import tensorflow as tf
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
                  num_nodes,
                  buffLen_in,
                  numVars_in = 20,):
    #Number of ouput classes
    num_classes = 3
    filter_height = 20 #numVars
    buffLen = tf.Variable(buffLen_in, name='buffLen') #declared to import when using saved model
    numVars = tf.Variable(numVars_in, name='numVars') #declared to import when using saved model
    #Other parameters
    learning_rate = 0.0001
    '''Placeholder variables'''
    
    #Input 'images'
    X = tf.placeholder(tf.float32, shape=[None, numVars_in, buffLen_in, 1], name='X')
    #output[#,20,90,1][numsamples, numVars, buffLen, 1]
    Y = tf.placeholder(tf.float32, shape=[None,num_classes], name='Y')
    #output[#,3][numsamples,num_classes]
    Y_cls = tf.argmax(Y, axis=1, name='Y_cls')

    '''Create convolution layer'''
    c1, w1 = new_conv_layer(X, 1, filter_width, filter_height, num_filters, pool_height, pool_width)
    
    '''Create flattened layer'''
    flat, num_features = flatten_layer(c1)
    
    '''Create fully-connected layer'''
    f1 = new_fc_layer(flat, num_features, num_nodes)
    
    '''Create fully-connected output layer'''
    f2 = new_logits(f1, num_nodes, num_classes)
    
    '''Create softmax output'''
    y_ = tf.nn.softmax(f2, name='y_')
    #Produces index of largest element
    y_pred_cls = tf.argmax(y_, axis=1,name='y_pred_cls')
    
    '''Cost function and optimizer'''
    # This is where the most improvement could come from
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=f2, labels=Y,name='cross_entropy')
    cost = tf.reduce_mean(cross_entropy, name='cost') #this is the mean of the cross entropy of all the image classifications
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    '''Performance Measures'''
    correct_prediction = tf.equal(y_pred_cls, Y_cls,name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    #shockingly this actually works as an accuracy measure when you are using 1 and 0
    return X, Y, Y_cls, y_, y_pred_cls, cross_entropy, cost, optimizer, correct_prediction, accuracy 


def plot_confusion_matrix(cls_pred, test_y_cls):
    # cls_pred is an array of the predicted class-number forall images in the test-set.
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=test_y_cls, y_pred=cls_pred)
    print(cm)
    sn.heatmap(cm, annot=True, fmt='g',annot_kws={"size": 30}, 
               square=True, xticklabels=['Sit','Stand','Walk'],
               yticklabels=['Sit','Stand','Walk'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()