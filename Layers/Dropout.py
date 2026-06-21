from Layers import Base
import numpy as np

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.trainable = False

    def forward(self, X):
        if self.testing_phase:
            return X
        self.mask = np.random.rand(*X.shape) < self.probability
        return X * self.mask / self.probability

    def backward(self, dL_dy):
        return dL_dy * self.mask / self.probability