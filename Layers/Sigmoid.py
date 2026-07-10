import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        self.X = 1 / (1 + np.exp(-X))
        return self.X

    def backward(self,dL_dy):
        dL_dx = np.multiply(self.X, 1-self.X)
        return dL_dy * dL_dx