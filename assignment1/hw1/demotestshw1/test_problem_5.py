import os
import numpy as np
import pickle
import json
from hw1 import hw1
from helpers.helpers import *

def test_problem_5s():
	STARTING_WEIGHTS_PATH = "fixtures/start_weights_problem5s"
	ENDING_WEIGHTS_PATH = "fixtures/final_weights_problem5s"
	STARTING_BIAS_PATH = "fixtures/start_bias_problem5s"
	ENDING_BIAS_PATH = "fixtures/final_bias_problem5s"
	PARAMS_PATH = "fixtures/problem5s.params.json"

	params = json.loads(open(PARAMS_PATH, "r").read())

	(X, Y)= hw1.load_data("data/digitstrain.txt")
	inputWeights = pickle.load(open(STARTING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)
	finalWeights = pickle.load(open(ENDING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)

	inputBias = pickle.load(open(STARTING_BIAS_PATH, "rb"), **PICKLE_KWARGS)
	finalBias = pickle.load(open(ENDING_BIAS_PATH, "rb"), **PICKLE_KWARGS)

	weightsToTest, biasToTest = hw1.update_weights_double_layer_act(X, Y, inputWeights, inputBias, float(params["LEARNING_RATE"]), params["ACTIVATION"])

	assert(isAllClose(finalWeights, weightsToTest))
	assert(isAllClose(finalBias, biasToTest))


def test_problem_5t():
	STARTING_WEIGHTS_PATH = "fixtures/start_weights_problem5t"
	ENDING_WEIGHTS_PATH = "fixtures/final_weights_problem5t"
	STARTING_BIAS_PATH = "fixtures/start_bias_problem5t"
	ENDING_BIAS_PATH = "fixtures/final_bias_problem5t"
	PARAMS_PATH = "fixtures/problem5t.params.json"

	params = json.loads(open(PARAMS_PATH, "r").read())

	(X, Y)= hw1.load_data("data/digitstrain.txt")
	inputWeights = pickle.load(open(STARTING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)
	finalWeights = pickle.load(open(ENDING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)

	inputBias = pickle.load(open(STARTING_BIAS_PATH, "rb"), **PICKLE_KWARGS)
	finalBias = pickle.load(open(ENDING_BIAS_PATH, "rb"), **PICKLE_KWARGS)

	weightsToTest, biasToTest = hw1.update_weights_double_layer_act(X, Y, inputWeights, inputBias, float(params["LEARNING_RATE"]), params["ACTIVATION"])

	assert(isAllClose(finalWeights, weightsToTest, tol=0.015))
	assert(isAllClose(finalBias, biasToTest, tol=0.015))

def test_problem_5r():
	STARTING_WEIGHTS_PATH = "fixtures/start_weights_problem5r"
	ENDING_WEIGHTS_PATH = "fixtures/final_weights_problem5r"
	STARTING_BIAS_PATH = "fixtures/start_bias_problem5r"
	ENDING_BIAS_PATH = "fixtures/final_bias_problem5r"
	PARAMS_PATH = "fixtures/problem5r.params.json"

	params = json.loads(open(PARAMS_PATH, "r").read())

	(X, Y)= hw1.load_data("data/digitstrain.txt")
	inputWeights = pickle.load(open(STARTING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)
	finalWeights = pickle.load(open(ENDING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)

	inputBias = pickle.load(open(STARTING_BIAS_PATH, "rb"), **PICKLE_KWARGS)
	finalBias = pickle.load(open(ENDING_BIAS_PATH, "rb"), **PICKLE_KWARGS)

	weightsToTest, biasToTest = hw1.update_weights_double_layer_act(X, Y, inputWeights, inputBias, float(params["LEARNING_RATE"]), params["ACTIVATION"])

	assert(isAllClose(finalWeights, weightsToTest))
	assert(isAllClose(finalBias, biasToTest))

