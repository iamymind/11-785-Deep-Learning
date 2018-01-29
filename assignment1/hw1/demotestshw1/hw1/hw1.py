#HANDLE YOUR IMPORTS HERE
import sys
import argparse
import numpy
import math
import pdb
import random
import pickle
import numpy as np
from scipy.special import expit

def softmax(z):
	"""compute softmax of NxD matrix"""
	exp_z = np.exp(z-np.max(z,axis=0))
	return exp_z/np.sum(exp_z,axis=1,keepdims=True)

#PROBLEM 1
def load_data(filePath):
	X = []
	Y = []
	with open(filePath,'r') as file:
        	for line in file:
                	dataPoint = list(map(float,line.split(',')))
                	X.append(dataPoint[0:len(dataPoint)-1])
                	Y.append(int(dataPoint[-1]))    
	X = np.array(X)
	Y = np.array(Y)

	return X, Y

#PROBLEM 2
def update_weights_perceptron(X, Y, weights, bias, lr):
    
	"""
	update the weight and bias of a NN with 0 hidden layer
	and softmax output.
	Divergence measure: cross entropy
	"""

	updated_weights = [] 
	updated_bias = []

	T = Y.shape[0]

	Z1 = np.dot(X,weights[0]) + bias[0]
	exp_Z1 = np.exp(Z1)
	
	output = exp_Z1/np.sum(exp_Z1,axis=1,keepdims=True)
	loss   = np.sum(-np.log(output[[range(T),Y]]))/T

	gradDivZ1 = output
	gradDivZ1[[range(T),Y]] -= 1

	gradDivZ1 /= T
	gradb1 = np.sum(gradDivZ1,axis=0,keepdims=True)
	gradW1 = np.dot(X.T,gradDivZ1)
	updated_weights.append(weights[0]-lr*gradW1)
	updated_bias.append(bias[0]-lr*gradb1)

	return updated_weights, updated_bias

#PROBLEM 3
def update_weights_single_layer(X, Y, weights, bias, lr):

	#INSERT YOUR CODE HERE    

	"""
	update the weight and bias of a NN with a single sigmoid activation layer
	and softmax output.eg. [ 784 > 100 > 10]
	Divergence measure: cross entropy
	"""

	#print("New")
	updated_weights = [] 
	updated_bias = []


	T = len(Y)#batch size
	#forward pass
	Z1 = np.dot(X,weights[0]) + bias[0]
	Y1 = expit(Z1)

	Z2 = np.dot(Y1,weights[1]) + bias[1]
	#Y2 = expit(Z2)
        #applying softmax
	output = softmax(Z2) 
	#computing loss
	loss   = np.sum(-np.log(output[[range(T),Y]]))/T
        #gradient with respect to softmax input
	gradDivZ2 = output
	gradDivZ2[[range(T),Y]] -= 1
	gradDivZ2 /= T

	#gradient with respect to output layer weight and bias 	
	gradb2 = np.sum(gradDivZ2,axis=0,keepdims=False)
	gradW2 = np.dot(Y1.T,gradDivZ2)
	#update output layer weight and bias
	updated_weights.insert(1,weights[1]-lr*gradW2)
	updated_bias.insert(1,bias[1]-lr*gradb2)
	

	gradDivY1 = np.dot(gradDivZ2,weights[1].T)
	sigmoidJacobian1 = Y1*(1-Y1)
	gradDivZ1 = sigmoidJacobian1*gradDivY1#
	gradb1 = np.sum(gradDivZ1,axis=0,keepdims=False)
	gradW1 = np.dot(np.array(X).T,gradDivZ1)

	updated_weights.insert(0,weights[0]-lr*gradW1)
	updated_bias.insert(0,bias[0]-lr*gradb1)


	return updated_weights, updated_bias

#PROBLEM 4
def update_weights_double_layer(X, Y, weights, bias, lr):

	#I intemidiate activations
	I  = [X]
	T = len(Y)#batch size
	#forward pass
	N = len(bias)
	updated_weights = [None]*N 
	updated_bias = [None]*N

	for k in range(1,N):

		Z = np.dot(I[k-1],weights[k-1]) + bias[k-1]
		Y_ = expit(Z)
		I.append(Y_)

	
	#last layer 
	ZN = np.dot(I[-1],weights[-1]) + bias[-1]
	#softmax
	output = softmax(ZN)
	
	#computing loss	
	loss   = np.sum(-np.log(output[[range(T),Y]]))/T
        #gradient with respect to softmax input
	
	gradDivYN = output
	gradDivYN[[range(T),Y]] -= 1
	gradDivYN /= T
	gradDivY = gradDivYN


	for k in reversed(range(1,N+1)):
		jacobian = None

		if k == N:
			jacobian = np.ones_like(gradDivY)
		else:
			jacobian = I[k]*(1-I[k])

		gradDivZ = jacobian*gradDivY#

		gradDivY = np.dot(gradDivZ,weights[k-1].T)
		
		gradb = np.sum(gradDivZ,axis=0,keepdims=False)
		gradW = np.dot(I[k-1].T,gradDivZ)

		updated_weights[k-1] = weights[k-1]-lr*gradW
		updated_bias[k-1]    = bias[k-1]-lr*gradb

	return updated_weights, updated_bias
#PROBLEM 5

def update_weights_double_layer_act(X, Y, weights, bias, lr, activation):

	I  = [X]
	T = len(Y)#batch size
	N = len(bias)
	updated_weights = [None]*N 
	updated_bias = [None]*N
	Z = [None]

	for k in range(1,N):

		Z_ = np.dot(I[k-1],weights[k-1]) + bias[k-1]
		Y_ = None
		if activation == 'sigmoid':
			Y_ = expit(Z_)
		if activation == 'tanh':
			Y_ = np.tanh(Z_)
		if activation == 'relu':
			Y_ = np.maximum(0,Z_)
		Z.append(Z_)
		I.append(Y_)
		
	#last layer 
	ZN = np.dot(I[-1],weights[-1]) + bias[-1]
	#softmax
	output = softmax(ZN)	
	#computing loss	
	loss   = np.sum(-np.log(output[[range(T),Y]]))/T
	
	gradDivYN = output
	gradDivYN[[range(T),Y]] -= 1
	gradDivYN /= T
	gradDivY = gradDivYN

	for k in reversed(range(1,N+1)):

		gradDivZ = None

		if k == N:

			gradDivZ = gradDivY#

		else:
			if activation == 'sigmoid':
				jacobian = I[k]*(1-I[k])
			if activation == 'tanh':
				jacobian = 1- np.power(I[k],2)			
			if activation == 'relu':
				jacobian = Z[k]
				jacobian[Z[k]<=0] = 0
				jacobian[Z[k]>0]  = 1

			gradDivZ   =  jacobian*gradDivY

		gradDivY = np.dot(gradDivZ,weights[k-1].T)		
		gradb = np.sum(gradDivZ,axis=0,keepdims=False)
		gradW = np.dot(I[k-1].T,gradDivZ)

		updated_weights[k-1] = weights[k-1]-lr*gradW
		updated_bias[k-1]    = bias[k-1]-lr*gradb


	return updated_weights, updated_bias


#PROBLEM 6
def update_weights_double_layer_act_mom(X, Y, weights, bias, lr, activation, momentum, epochs):

	I  = [X]
	T = len(Y)#batch size
	N = len(bias)#number of layers 

	Z = [None]
	delta_gradW = []
	delta_gradb = []

	updated_weights = weights
	updated_bias    = bias

	for l in range(N):#for every layer

		delta_gradW.append(np.zeros_like(updated_weights[l]))
		delta_gradb.append(np.zeros_like(updated_bias[l]))

	for epoch in range(epochs):

		for k in range(1,N):

			Z_ = np.dot(I[k-1],updated_weights[k-1]) + updated_bias[k-1]
			Y_ = None
			if activation == 'sigmoid':
				Y_ = expit(Z_)
			if activation == 'tanh':
				Y_ = np.tanh(Z_)
			if activation == 'relu':
				Y_ = np.maximum(0,Z_)
			Z.append(Z_)
			I.append(Y_)
		
		#last layer 
		ZN = np.dot(I[-1],updated_weights[-1]) + updated_bias[-1]
		#softmax
		output = softmax(ZN)	
		#computing loss	
		loss   = np.sum(-np.log(output[[range(T),Y]]))/T

		gradDivYN = output
		gradDivYN[[range(T),Y]] -= 1
		gradDivYN /= T
		gradDivY = gradDivYN

		for k in reversed(range(1,N+1)):

			gradDivZ = None

			if k == N:

				gradDivZ = gradDivY#

			else:
				if activation == 'sigmoid':
					jacobian = I[k]*(1-I[k])
				if activation == 'tanh':
					jacobian = 1- np.power(I[k],2)			
				if activation == 'relu':
					jacobian = Z[k]
					jacobian[Z[k]<=0] = 0
					jacobian[Z[k]>0]  = 1

				gradDivZ   =  jacobian*gradDivY

			gradDivY = np.dot(gradDivZ,updated_weights[k-1].T)
		
			gradb = np.sum(gradDivZ,axis=0,keepdims=False)
			gradW = np.dot(I[k-1].T,gradDivZ)

			delta_gradW[k-1] = momentum*delta_gradW[k-1] - lr*gradW
			delta_gradb[k-1] = momentum*delta_gradb[k-1] - lr*gradb

			updated_weights[k-1] += delta_gradW[k-1] 
			updated_bias[k-1]    += delta_gradb[k-1]
		print(delta_gradW[0][500])
	return updated_weights, updated_bias

