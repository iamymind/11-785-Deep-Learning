import os
import pickle
from hw1 import hw1
from helpers.helpers import *

def test_problem_1():
    REF_XTRAIN = "fixtures/X_expected"
    REF_YTRAIN = "fixtures/Y_expected"

    (xtrain, ytrain) = hw1.load_data("data/digitstest.txt")

    ref_xtrain = pickle.load(open(REF_XTRAIN, "rb"), **PICKLE_KWARGS)
    ref_ytrain = pickle.load(open(REF_YTRAIN, "rb"), **PICKLE_KWARGS)

    assert(ref_xtrain.shape == xtrain.shape)
    assert(ref_ytrain.shape == ytrain.shape)
    assert(isAllClose(ref_ytrain, ytrain))
    assert(isAllClose(ref_xtrain, xtrain))
    



    



