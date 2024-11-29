import numpy as np
from .variable import Variable

def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x