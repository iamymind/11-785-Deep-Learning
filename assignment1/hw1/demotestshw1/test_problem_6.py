import os
import numpy as np
import pickle
import json
from hw1 import hw1
from helpers.helpers import *
from scipy.special import expit

def softmax(z):
	"""compute softmax of NxD matrix"""
	exp_z = np.exp(z-np.max(z,axis=0))
	return exp_z/np.sum(exp_z,axis=1,keepdims=True)

def test_problem_6():
	STARTING_WEIGHTS_PATH = "fixtures/start_weights_problem6"
	ENDING_WEIGHTS_PATH = "fixtures/final_weights_problem6"
	STARTING_BIAS_PATH = "fixtures/start_bias_problem6"
	ENDING_BIAS_PATH = "fixtures/final_bias_problem6"
	PARAMS_PATH = "fixtures/problem6.params.json"

	params = json.loads(open(PARAMS_PATH, "r").read())

	(X, Y)= hw1.load_data("data/digitstrain.txt")
	inputWeights = pickle.load(open(STARTING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)
	finalWeights = pickle.load(open(ENDING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)

	inputBias = pickle.load(open(STARTING_BIAS_PATH, "rb"), **PICKLE_KWARGS)
	finalBias = pickle.load(open(ENDING_BIAS_PATH, "rb"), **PICKLE_KWARGS)

	inputLoss,inputAccuracy = forward_and_compute_loss(X,Y,inputWeights,inputBias,float(params["LEARNING_RATE"]), params["ACTIVATION"])

	weightsToTest, biasToTest = hw1.update_weights_double_layer_act_mom(X, Y, inputWeights, inputBias, float(params["LEARNING_RATE"]), params["ACTIVATION"], float(params["MOMENTUM"]), int(params["EPOCH_COUNT"]))
	
	correctLoss,correctAccuracy = forward_and_compute_loss(X,Y,finalWeights,finalBias,float(params["LEARNING_RATE"]), params["ACTIVATION"])

	myLoss,myAccuracy = forward_and_compute_loss(X,Y,weightsToTest,biasToTest,float(params["LEARNING_RATE"]), params["ACTIVATION"])

	print("inputLoss:   ",inputLoss,  " inputAccuracy:   ",inputAccuracy)
	print("correctLoss: ",correctLoss," correctAccuracy: ",correctAccuracy)
	print("myLoss:      ",myLoss,     " myAccuracy:      ",myAccuracy)
	
	assert(isAllClose(finalWeights, weightsToTest))
	assert(isAllClose(finalBias, biasToTest))

def forward_and_compute_loss(X,Y,weights,bias,lr,activation):

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
	accuracy = np.mean(np.argmax(output, axis=1)==Y)*100

	return loss, accuracy
