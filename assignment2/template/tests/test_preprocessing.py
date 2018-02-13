import os
import unittest.mock
import warnings

import numpy as np
import pytest

from hw2 import preprocessing as P


def load_fixture(filename):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fixtures', filename)
    return np.load(path)


def assert_close(a, b, rtol=1.e-5, atol=1e-6):
    assert np.all(np.equal(a.shape, b.shape))
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        warnings.warn("Max difference between arrays: {}".format(np.max(np.abs(a - b))))
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def test_sample_zero_mean():
    fixture = load_fixture('sample_zero_mean.npz')
    output = P.sample_zero_mean(fixture['input'])
    assert_close(output, fixture['output'])


def test_gcn():
    fixture = load_fixture('gcn.npz')
    output = P.gcn(fixture['input'])
    assert_close(output, fixture['output'])


def test_feature_zero_mean():
    fixture = load_fixture('feature_zero_mean.npz')
    o0, o1 = P.feature_zero_mean(fixture['x0'], fixture['x1'])
    assert_close(o0, fixture['o0'])
    assert_close(o1, fixture['o1'])


def test_zca():
    fixture = load_fixture('zca.npz')
    with unittest.mock.patch('numpy.linalg') as linalg:
        linalg.svd = unittest.mock.MagicMock(return_value=(fixture['U'], fixture['S'], fixture['V']))
        o0, o1 = P.zca(fixture['x0'], fixture['x1'])
        assert linalg.svd.called
        assert_close(linalg.svd.call_args[0][0], fixture['sigma'], atol=0.1, rtol=0.01)
    assert_close(o0, fixture['o0'], atol=1e-2)
    assert_close(o1, fixture['o1'], atol=1e-2)


def test_cifar_10_preprocess():
    fixture = load_fixture('cifar_10_preprocess.npz')
    with unittest.mock.patch('numpy.linalg') as linalg:
        linalg.svd = unittest.mock.MagicMock(return_value=(fixture['U'], fixture['S'], fixture['V']))
        o0, o1 = P.cifar_10_preprocess(fixture['x0'], fixture['x1'], 10)
        assert linalg.svd.called
        assert_close(linalg.svd.call_args[0][0], fixture['sigma'], atol=0.1, rtol=0.01)
    assert_close(o0, fixture['o0'], atol=1e-2)
    assert_close(o1, fixture['o1'], atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '--duration=0'])
