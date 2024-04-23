import weakref
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data  # 데이터 ndarray
        self.name = name
        self.grad = None  # 미분값을 저장하는 변수 ndarray
        self.creator = None  # 변수의 창조자(creator)를 기억하는 변수
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대를 기록한다(부모 세대 + 1)

    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input  # 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad)  # 함수의 backward 메서드를 호출한다.
    #         x.backward()  # 하나 앞 변수의 backward 메서드를 호출한다.

    def backward(self, retain_grad=False):
        # Variable 클래스의 backward 메서드는 반복문을 사용해 구현한다.
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # funcs 리스트에는 Variable 인스턴스의 creator가 차례로 추가된다.
        funcs = []
        # seen_set은 중복을 방지하기 위한 집합(set)이다.        
        seen_set = set()

        # funcs 리스트에 creator를 추가하고, 그 함수를 func이라는 변수에 저장한다.
        def add_func(f):
            if f not in seen_set:
                funcs.append(f) # 중복을 방지하기 위해 funcs 리스트에 추가한다.
                seen_set.add(f) # 추가할 때 집합에도 추가한다.
                funcs.sort(key=lambda x: x.generation) # 세대 순으로 정렬한다.

        add_func(self.creator) # 첫 번째 함수를 추가한다.

        # funcs 리스트에 원소가 있는 동안 반복하며 역전파를 수행한다.
        while funcs:
            f = funcs.pop() # funcs 리스트의 마지막 원소를 가져온다.
            gys = [output().grad for output in f.outputs]  # 출력변수인 outputs의 grad를 리스트에 담는다.
            gxs = f.backward(*gys)  # f의 역전파를 호출한다. f는 Function 클래스의 인스턴스이다.
            if not isinstance(gxs, tuple): # gxs가 튜플이 아닌 경우 추가 지원
                gxs = (gxs,) # 튜플로 변환한다.

            for x, gx in zip(f.inputs, gxs): # 역전파로 전파되는 미분값을 Variable의 grad 필드에 담는다.
                if x.grad is None: # 미분값을 처음 설정할 때는 gx를 그대로 대입한다.
                    x.grad = gx # 미분값을 저장한다.
                else:         # 그렇지 않으면 기존에 저장된 미분값에 gx를 더한다.
                    x.grad = x.grad + gx 

                if x.creator is not None: # 한 번 더 역전파를 하기 위해 add_func 함수를 호출한다.
                    add_func(x.creator)

            if not retain_grad: # retain_grad가 False인 경우 중간 변수의 미분값을 모두 None으로 설정한다.
                for y in f.outputs:
                    y().grad = None 

    def cleargrad(self):
        self.grad = None
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'
    