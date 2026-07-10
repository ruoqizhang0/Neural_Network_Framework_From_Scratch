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
        self.regularizer = None

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

        if self.regularizer is not None:
            self.dL_dW += self.regularizer.calculate_gradient(self.weights)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.dL_dW)

        return self.dL_dx

    def calculate_regularization_loss(self):

        if self.regularizer is None:
            return 0

        return self.regularizer.norm(self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.regularizer = optimizer.regularizer

    @property
    def gradient_weights(self):
        return self.dL_dW

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size
        fan_out = self.output_size

        self.weights[1:, :] = weights_initializer.initialize(
            self.weights[1:, :].shape,
            fan_in,
            fan_out
        )

        self.weights[0, :] = bias_initializer.initialize(
            self.weights[0, :].shape,
            fan_in,
            fan_out
        )
