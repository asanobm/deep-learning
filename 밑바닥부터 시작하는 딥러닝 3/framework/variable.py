class Variable:
    def __init__(self, data):
        self.data = data  # 데이터 ndarray
        self.grad = None  # 미분값을 저장하는 변수 ndarray
        self.creator = None  # 변수의 창조자(creator)를 기억하는 변수

    def set_creator(self, func):
        self.creator = func
