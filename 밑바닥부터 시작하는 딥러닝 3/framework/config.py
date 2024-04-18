import contextlib

class Config:
  enable_backprop = True
  
@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)
    
def no_grad():
  return using_config('enable_backprop', False)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)