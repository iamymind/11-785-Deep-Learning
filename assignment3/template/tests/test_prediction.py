import os
import warnings

import numpy as np
import pytest

import hw3.prediction


def fixture_path(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fixtures', name)


def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm


def test_prediction():
    fixture = np.load(fixture_path('prediction.npz'))
    inp = fixture['inp']
    targ = fixture['out']
    print('targ: ', targ.shape)
    out = hw3.prediction.prediction(inp.copy())
    assert out.shape[0] == targ.shape[0]
    vocab = np.load(fixture_path('vocab.npy'))
    assert out.shape[1] == vocab.shape[0]
    out = log_softmax(out, 1)
    nlls = out[np.arange(out.shape[0]), targ]
    nll = -np.mean(nlls)
    warnings.warn("Your mean NLL for predicting a single word: {}".format(nll))
    assert nll < 5.5


if __name__ == '__main__':
    pytest.main([__file__])
