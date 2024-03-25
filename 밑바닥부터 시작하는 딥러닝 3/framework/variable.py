class Variable:
    def __init__(self, data):
        self.data = data  # 데이터 ndarray
        self.grad = None  # 미분값을 저장하는 변수 ndarray
