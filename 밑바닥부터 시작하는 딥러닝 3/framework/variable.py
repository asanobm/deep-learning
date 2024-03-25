import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data  # 데이터 ndarray
        self.grad = None  # 미분값을 저장하는 변수 ndarray
        self.creator = None  # 변수의 창조자(creator)를 기억하는 변수

    def set_creator(self, func):
        self.creator = func

    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input  # 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad)  # 함수의 backward 메서드를 호출한다.
    #         x.backward()  # 하나 앞 변수의 backward 메서드를 호출한다.

    def backward(self):
        """재귀에서 반복문으로 변경"""

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = [self.creator]

        while functions:
            f = functions.pop()  # 함수를 가져온다.
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)  # backward 메서드를 호출한다.

            if x.creator is not None:
                functions.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.
