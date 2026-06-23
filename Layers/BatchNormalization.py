import copy

from Layers import Base
import numpy as np
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.mean = None
        self.var = None
        self.alpha = 0.8
        self.eps = 1e-12
        self._optimizer = None
        self.weights = np.ones(channels)
        self.bias = np.zeros(channels)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, X):
        self.X_shape = X.shape
        X_dim = X.ndim

        if X.ndim == 4:
            X = self.reformat(X)

        if not self.testing_phase:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            if self.mean is None:  # the first batch value
                self.mean = mean
                self.var = var
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * mean
                self.var = self.alpha * self.var + (1 - self.alpha) * var

            self.batch_mean = mean
            self.batch_var = var
        else:
            mean = self.mean
            var = self.var

        self.X = X
        self.X_norm = (X - mean) / np.sqrt(var + self.eps)

        Y = self.X_norm * self.weights + self.bias

        if X_dim == 4:
            Y = self.reformat(Y)

        return Y

    def backward(self, dL_dy):
        dL_dy_ndim = dL_dy.ndim

        if dL_dy.ndim == 4:
            dL_dy = self.reformat(dL_dy)

        self._gradient_weights = np.sum(dL_dy * self.X_norm, axis=0)
        self._gradient_bias = np.sum(dL_dy, axis=0)

        dL_dx = compute_bn_gradients(
            dL_dy,
            self.X,
            self.weights,
            self.batch_mean,
            self.batch_var
        )

        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(
                self.weights, self._gradient_weights
            )
            self.bias = self._optimizer_bias.calculate_update(
                self.bias, self._gradient_bias
            )

        if dL_dy_ndim == 4:
            dL_dx = self.reformat(dL_dx)

        return dL_dx

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.original_shape = tensor.shape
            b, c, m, n = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, c)

        elif tensor.ndim == 2:
            b, c, m, n = self.original_shape
            return tensor.reshape(b, m, n, c).transpose(0, 3, 1, 2)