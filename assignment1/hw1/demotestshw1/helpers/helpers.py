import numpy as np
import six

PICKLE_KWARGS = {'encoding':'latin1'} if six.PY3 else {}

def isAllClose(a, b, tol=0.01):
  LIST_TYPE = type([])
  if(type(a) == LIST_TYPE or type(b) == LIST_TYPE):
    for i, j in zip(a, b):
      if(not np.allclose(i, j, atol=tol)):
        return False
    return True
  return np.allclose(a, b, atol=tol)
