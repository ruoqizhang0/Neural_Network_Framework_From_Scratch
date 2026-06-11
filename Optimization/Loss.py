import numpy as np
from Layers import SoftMax

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_k,y):
        self.epsilon = np.finfo(float).eps
        self.y_k = y_k
        loss = np.sum( -y * np.log(self.y_k + self.epsilon))
        return loss

    def backward(self,y):
        return -y / (self.y_k + self.epsilon)