import numpy as np
from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        self.X = X
        return np.maximum(0,X)

    def backward(self,dL_dy):
        dReLU = (self.X > 0).astype(float)
        return dL_dy * dReLU