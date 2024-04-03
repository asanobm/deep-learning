"""Function base class
"""
import numpy as np
from .variable import Variable


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs] # Variable 인스턴스로부터 데이터를 꺼낸다.
        ys = self.forward(xs) # forward 메서드에서 구체적인 계산을 수행한다.
        outputs = [Variable(as_array(y)) for y in ys] # 계산된 데이터를 Variable 인스턴스로 다시 감싼다.
        
        for output in outputs:
            output.set_creator(self) # 원산지 표시를 한다.
            
        self.inputs = inputs  # 입력 변수를 기억(보관)한다.
        self.outputs = outputs  # 출력 변수를 저장한다.
        return outputs

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


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)
