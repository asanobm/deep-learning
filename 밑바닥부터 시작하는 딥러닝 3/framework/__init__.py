from .function import *
from .variable import Variable
from .config import *
from .common import *

Variable.__add__ = add
Variable.__mul__ = mul
Variable.__radd__ = add
Variable.__rmul__ = mul