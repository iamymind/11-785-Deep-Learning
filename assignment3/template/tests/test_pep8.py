import os

import pytest

import hw3


def test_pep8():
    path = os.path.dirname(os.path.dirname(hw3.__file__))
    ret = pytest.main(['--pep8', '-m', 'pep8', path])
    assert ret == 0


if __name__ == '__main__':
    pytest.main([__file__, '--duration=0'])
