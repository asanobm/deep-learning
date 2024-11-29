import numpy as np
import weakref
from framework.variable import Variable
from framework.config import Config
from framework.common import as_array, as_variable


class Function:
    def __call__(self, *inputs):
        # 21. 연산자 오버로드(2)
        inputs = [as_variable(x) for x in inputs] # Variable로 변환한다.
        xs = [x.data for x in inputs] # Variable 인스턴스로부터 데이터를 꺼낸다.
        ys = self.forward(*xs) # forward 메서드에서 구체적인 계산을 수행한다.
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # 계산된 데이터를 Variable 인스턴스로 다시 감싼다.
        if Config.enable_backprop: # 역전파가 필요한 경우
            self.generation = max([x.generation for x in inputs]) # 입력 변수가 둘 이상일 때 가장 큰 generation을 선택한다.
            for output in outputs:
                output.set_creator(self) # 원산지 표시를 한다.
                
            self.inputs = inputs  # 입력 변수를 기억(보관)한다.
            # self.outputs = outputs  # 출력 변수를 저장한다.
            self.outputs = [weakref.ref(output) for output in outputs] # 출력변수를 약한 참조로 가지기
        
        return outputs if len(outputs) > 1 else outputs[0] # 출력이 하나라면 첫 번째 요소를 반환한다.

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
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    return Square()(x)


class Exp(Function):
    """지수 함수를 계산한다.
    """

    def forward(self, x):
        """
        $$y = \exp(x)$$
        """
        return np.exp(x)

    def backward(self, gy):
        """
        $$\frac{dy}{dx} = \exp(x)$$
        """
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def exp(x):
    return Exp()(x)
    
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)  # 반환값을 튜플로 묶는다.
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)