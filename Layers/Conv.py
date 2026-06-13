import copy

import numpy as np
from scipy import signal

from Layers import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None

        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

        self.X = None
        self.X_shape = None

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


    def forward(self, X):
        self.X_shape = X.shape
        self.X = X
        self.batch_size = X.shape[0]
        self.spatial_shape = X.shape[2:]

        Y_spatial_shape = tuple(
            int(np.ceil(self.spatial_shape[i] / self.stride_shape[i]))
            for i in range(len(self.spatial_shape))
        )
        Y = np.zeros((self.batch_size, self.num_kernels, *Y_spatial_shape))
        for b in range(self.batch_size):
            for k in range(self.num_kernels):
                X_spatial_shape = np.zeros(self.spatial_shape)
                for c in range(X.shape[1]):
                    X_spatial_shape += signal.correlate(X[b, c], self.weights[k, c], mode="same")
                X_spatial_shape += self.bias[k]

                if len(self.spatial_shape) == 1:
                    Y[b, k] = X_spatial_shape[::self.stride_shape[0]]
                else:
                    Y[b, k] = X_spatial_shape[
                        ::self.stride_shape[0],
                        ::self.stride_shape[1]
                    ]
        return Y

    def backward(self, dL_dY):

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias= np.zeros_like(self.bias)
        dL_dX = np.zeros_like(self.X)

        for b in range(self.batch_size):
            for k in range(self.num_kernels):
                ## pad the dL_dY
                upsampled_error = np.zeros(self.spatial_shape)
                if len(self.spatial_shape) == 1:
                    upsampled_error[::self.stride_shape[0]] = dL_dY[b,k]
                else:
                    upsampled_error[::self.stride_shape[0],
                    ::self.stride_shape[1]] = dL_dY[b,k]

                self._gradient_bias[k] += np.sum(dL_dY[b, k]) #dL_dB

                for c in range(self.X_shape[1]):
                    ##pad input to calculate dL_dW
                    kernel_shape = self.weights[k,c].shape
                    if len(self.spatial_shape) == 1:
                        padded_X = np.pad(self.X[b, c], (kernel_shape[0]//2, (kernel_shape[0]-1)//2), mode="constant")
                    else:
                        padded_X = np.pad(self.X[b, c], ((kernel_shape[0]//2, (kernel_shape[0]-1)//2),
                                                        (kernel_shape[1]//2, (kernel_shape[1]-1)//2)), mode="constant")

                    self._gradient_weights[k,c] += signal.correlate(padded_X, upsampled_error, mode="valid" ) #dL_dW

                    dL_dX[b, c] += signal.convolve(upsampled_error, self.weights[k, c], mode="same")

        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(
                self.weights,
                self._gradient_weights
            )
            self.bias = self._optimizer_bias.calculate_update(
                self.bias,
                self._gradient_bias
            )

        return dL_dX

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])

        self.weights = weights_initializer.initialize(
            self.weights.shape,
            fan_in,
            fan_out
        )

        self.bias = bias_initializer.initialize(
            self.bias.shape,
            fan_in,
            fan_out
        )