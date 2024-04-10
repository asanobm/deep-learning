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
        """재귀에서 반복문으로 변경"""

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # functions = [self.creator]
        funcs = []
        
        seen_set = set() # 중복을 방지하기 위한 집합(set) 자료구조
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 가져온다.
            # x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            # x.grad = f.backward(y.grad)  # backward 메서드를 호출한다.

            # if x.creator is not None:
            #     functions.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.
            
            # gys = [output.grad for output in f.outputs] # 출력변수인 output에서 출력변수의 미분을 가져온다.
            gys = [output().grad for output in f.outputs] # 출력변수인 output에서 출력변수의 미분을 가져온다.(output은 약한 참조이므로 ()를 붙여준다.)
            gxs = f.backward(*gys) # 역전파 호출
            if not isinstance(gxs, tuple): # 튜플이 아닌 경우 추가 지원
                gxs = (gxs,)
                
            for x, gx in zip(f.inputs, gxs): # 역전파로 전파되는 미분을 Variable의 인스턴스 변수 grad에 저장한다.
                # x.grad = gx # 같은 변수를 반복 사용할 경우 덮어쓰기 때문에 주의해야 한다.
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None: # 하나 앞의 함수를 리스트에 추가한다.
                    add_func(x.creator)
            if not retain_grad:
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