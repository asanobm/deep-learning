"""Function base class
"""
import numpy as np
from .variable import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input  # 입력 변수를 기억(보관)한다.
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    """입력값을 제곱한다."""

    def forward(self, x):
        """ $$ y = x^2 $$
        """
        return x ** 2

    def backward(self, gy):
        """$$\frac{dy}{dx} = 2x$$
        """
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """지수 함수를 계산한다.
    """

    def forward(self, x):
        """$$y = \exp(x)$$
        """
        return np.exp(x)

    def backward(self, gy):
        """$$\frac{dy}{dx} = \exp(x)$$
        """
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
