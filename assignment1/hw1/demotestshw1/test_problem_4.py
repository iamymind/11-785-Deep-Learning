import os
import numpy as np
import pickle
import json
from hw1 import hw1
from helpers.helpers import *


def test_problem_4():
	STARTING_WEIGHTS_PATH = "fixtures/start_weights_problem4"
	ENDING_WEIGHTS_PATH = "fixtures/final_weights_problem4"
	STARTING_BIAS_PATH = "fixtures/start_bias_problem4"
	ENDING_BIAS_PATH = "fixtures/final_bias_problem4"
	PARAMS_PATH = "fixtures/problem4.params.json"

	params = json.loads(open(PARAMS_PATH, "r").read())

	(X, Y)= hw1.load_data("data/digitstrain.txt")
	inputWeights = pickle.load(open(STARTING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)
	finalWeights = pickle.load(open(ENDING_WEIGHTS_PATH, "rb"), **PICKLE_KWARGS)

	inputBias = pickle.load(open(STARTING_BIAS_PATH, "rb"), **PICKLE_KWARGS)
	finalBias = pickle.load(open(ENDING_BIAS_PATH, "rb"), **PICKLE_KWARGS)

	weightsToTest, biasToTest = hw1.update_weights_double_layer(X, Y, inputWeights, inputBias, float(params["LEARNING_RATE"]))

	assert(isAllClose(finalWeights, weightsToTest))
	assert(isAllClose(finalBias, biasToTest))
