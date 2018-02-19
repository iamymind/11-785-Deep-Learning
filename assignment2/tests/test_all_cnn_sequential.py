import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

import hw2.all_cnn


def test_all_cnn_shape():
    model = hw2.all_cnn.all_cnn_module()
    input = Variable(torch.zeros(65, 3, 32, 32))
    output = model(input)
    assert output.size()[0] == 65
    assert output.size()[1] == 10


def assert_dropout(layer, p):
    assert isinstance(layer, nn.Dropout)
    assert layer.p == p


def assert_conv2d(layer, in_channels, out_channels, kernel_size, padding, stride):
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == in_channels
    assert layer.out_channels == out_channels
    assert layer.kernel_size[0] == kernel_size
    assert layer.padding[0] == padding
    assert layer.stride[0] == stride


def assert_relu(layer):
    assert isinstance(layer, nn.ReLU)


def assert_avg(layer, kernel_size):
    assert isinstance(layer, nn.AvgPool2d)
    assert _pair(layer.kernel_size)[0] == kernel_size


def assert_flatten(layer):
    assert isinstance(layer, hw2.all_cnn.Flatten)


def test_all_cnn_module():
    model = hw2.all_cnn.all_cnn_module()
    assert len(model) == 23

    assert_dropout(model[0], 0.2)
    assert_conv2d(model[1], 3, 96, 3, 1, 1)
    assert_relu(model[2])
    assert_conv2d(model[3], 96, 96, 3, 1, 1)
    assert_relu(model[4])
    assert_conv2d(model[5], 96, 96, 3, 1, 2)
    assert_relu(model[6])

    assert_dropout(model[7], 0.5)
    assert_conv2d(model[8], 96, 192, 3, 1, 1)
    assert_relu(model[9])
    assert_conv2d(model[10], 192, 192, 3, 1, 1)
    assert_relu(model[11])
    assert_conv2d(model[12], 192, 192, 3, 1, 2)
    assert_relu(model[13])

    assert_dropout(model[14], 0.5)
    assert_conv2d(model[15], 192, 192, 3, 0, 1)
    assert_relu(model[16])
    assert_conv2d(model[17], 192, 192, 1, 0, 1)
    assert_relu(model[18])
    assert_conv2d(model[19], 192, 10, 1, 0, 1)
    assert_relu(model[20])

    assert_avg(model[21], 6)
    assert_flatten(model[22])


def test_flatten_module():
    input = np.arange(36 * 17).reshape((36, 17))
    varin = Variable(torch.from_numpy(input))
    layer = hw2.all_cnn.Flatten()
    varout = layer(varin.view(36, 17, 1, 1))
    output = varout.data.numpy()
    assert np.all(np.equal(input.shape, output.shape))
    assert np.allclose(input, output)


if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '--duration=0'])
