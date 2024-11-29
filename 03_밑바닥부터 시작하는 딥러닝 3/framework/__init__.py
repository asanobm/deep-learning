from .function import *
from .variable import Variable
from .config import *
from .common import *
from .utils import *

Variable.__add__ = add
Variable.__mul__ = mul
Variable.__radd__ = add
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
