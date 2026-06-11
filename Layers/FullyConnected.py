from fontTools.feaLib import error

from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None

    def forward(self, X):
        batch_size = X.shape[0]
        bias = np.ones((batch_size, 1))
        X_bias = np.concatenate((bias, X), axis=1)
        self.X_bias = X_bias #X

        return np.dot(self.X_bias, self.weights)

    def backward(self, dL_dy):
        dy_dx = self.weights[1:]
        dy_dw = self.X_bias
        self.dL_dx = np.dot(dL_dy, dy_dx.T)
        self.dL_dW = np.dot(dy_dw.T, dL_dy)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.dL_dW)

        return self.dL_dx

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self.dL_dW