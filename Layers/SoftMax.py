import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        self.X = X
        self.X = self.X - np.max(self.X, axis=1, keepdims=True)
        exp = np.exp(self.X)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        self.y_k = exp / exp_sum
        return self.y_k
    def backward(self,dL_dy):
        self.dL_dy = dL_dy
        dL_dy = self.y_k*(self.dL_dy - np.sum(self.dL_dy * self.y_k, axis=1, keepdims=True))
        return dL_dy