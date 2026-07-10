import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        self.X = np.tanh(X)
        return self.X

    def backward(self,dL_dy):
        dL_dx = 1 - np.multiply(self.X, self.X)
        return dL_dy * dL_dx