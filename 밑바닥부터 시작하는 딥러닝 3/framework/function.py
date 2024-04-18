import numpy as np
import weakref
from framework.variable import Variable
from framework.config import Config


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
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
    
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)  # 반환값을 튜플로 묶는다.
    
    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)
