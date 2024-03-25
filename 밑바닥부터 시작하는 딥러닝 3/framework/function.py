"""Function base class
"""
import numpy as np
from .variable import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    """입력값을 제곱한다."""

    def forward(self, x):
        return x ** 2


class Exp(Function):
    """$y=e^x$를 계산한다.
    """

    def forward(self, x):
        return np.exp(x)
