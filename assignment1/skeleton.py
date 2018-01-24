import sys
import argparse
import numpy
import math
import pdb
import random
import pickle

def load_data(filePath):
    X = []
    Y = []
    with open(filePath,'r') as file:
        for line in file:
                dataPoint = list(map(float,line.split(',')))
                X.append(dataPoint[0:len(dataPoint)-1])
                Y.append(int(dataPoint[-1]))    
    return X, Y

def update_weights_perceptron(X, Y, weights, lr):
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_single_layer(X, Y, weights, lr):
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_single_layer_mean(X, Y, weights, lr):
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_double_layer(X, Y, weights, lr):
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_double_layer_batch(X, Y, weights, lr, batch_size):
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_double_layer_batch_act(X, Y, weights, lr, batch_size, activation):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'tanh':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'relu':
        pass
        #INSERT YOUR CODE HERE
    #INSERT YOUR CODE HERE
    return updated_weights

def update_weights_double_layer_batch_act_mom(X, Y, weights, lr, batch_size, activation, momentum):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'tanh':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'relu':
        pass
        #INSERT YOUR CODE HERE
    #INSERT YOUR CODE HERE
    return updated_weights

