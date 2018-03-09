import os
import warnings

import numpy as np
import pytest

import hw3.generation


def fixture_path(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fixtures', name)


def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm


def array_to_str(arr, vocab):
    return " ".join(vocab[a] for a in arr)


def test_generation():
    inp = np.load(fixture_path('generation.npy'))
    forward = 20
    n = inp.shape[0]
    pred = hw3.generation.generation(inp, forward)
    assert pred.shape[0] == n
    assert pred.shape[1] == forward
    vocab = np.load(fixture_path('vocab.npy'))
    for i in range(n):
        w1 = array_to_str(inp[i], vocab)
        w2 = array_to_str(pred[i], vocab)
        warnings.warn_explicit("Input | Output #{}: {} | {}".format(i, w1, w2), Warning, 'generated', i)

    warnings.warn(
        "Passing this test does not mean you will pass on autolab. Read the generated strings that " +
        " were printed and see if they seem reasonable.")


if __name__ == '__main__':
    pytest.main([__file__])
