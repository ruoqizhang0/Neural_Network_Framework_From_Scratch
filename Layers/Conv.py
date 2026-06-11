import copy

import numpy as np
from scipy import signal

from Layers import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernel):
        super().__init__()
        self.trainable = True

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        self.num_kernel = num_kernel

        self.weights = np.random.rand(self.num_kernel, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernel)

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
        self.X = X
        self.X_shape = X.shape
        self.batch_size = X.shape[0]
        self.spatial_shape = X.shape[2:]
        output_spatial_shape = tuple(
            int(np.ceil(self.spatial_shape[i] / self.stride_shape[i]))
            for i in range(len(self.spatial_shape))
        )

        X_out = np.zeros((self.batch_size, self.num_kernel, *output_spatial_shape))
        for b in range(self.batch_size):
            for k in range(self.num_kernel):
                X_spatial_shape = np.zeros(self.spatial_shape)
                for c in range(X.shape[1]):
                    X_spatial_shape += signal.correlate(X[b, c], self.weights[k, c], mode="same")
                X_spatial_shape += self.bias[k]

                if len(self.spatial_shape) == 1:
                    X_out[b, k] = X_spatial_shape[::self.stride_shape[0]]
                else:
                    X_out[b, k] = X_spatial_shape[
                        ::self.stride_shape[0],
                        ::self.stride_shape[1]
                    ]
        return X_out

    def backward(self, de_do):

        channels = self.X.shape[1]

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        gradient_input = np.zeros_like(self.X)

        for b in range(self.batch_size):
            for k in range(self.num_kernel):
                upsampled_error = np.zeros(self.spatial_shape)

                if len(self.spatial_shape) == 1:
                    upsampled_error[::self.stride_shape[0]] = de_do[b, k]
                else:
                    upsampled_error[
                        ::self.stride_shape[0],
                        ::self.stride_shape[1]
                    ] = de_do[b, k]

                self._gradient_bias[k] += np.sum(de_do[b, k])

                for c in range(channels):
                    kernel_shape = self.weights[k, c].shape

                    if len(self.spatial_shape) == 1:
                        pad_left = kernel_shape[0] // 2
                        pad_right = (kernel_shape[0] - 1) // 2

                        padded_input = np.pad(
                            self.X[b, c],
                            (pad_left, pad_right),
                            mode="constant"
                        )

                    else:
                        pad_top = kernel_shape[0] // 2
                        pad_bottom = (kernel_shape[0] - 1) // 2
                        pad_left = kernel_shape[1] // 2
                        pad_right = (kernel_shape[1] - 1) // 2

                        padded_input = np.pad(
                            self.X[b, c],
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode="constant"
                        )

                    self._gradient_weights[k, c] += signal.correlate(
                        padded_input,
                        upsampled_error,
                        mode="valid"
                    )

                    gradient_input[b, c] += signal.convolve(
                        upsampled_error,
                        self.weights[k, c],
                        mode="same"
                    )

        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(
                self.weights,
                self._gradient_weights
            )
            self.bias = self._optimizer_bias.calculate_update(
                self.bias,
                self._gradient_bias
            )

        return gradient_input


    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernel * np.prod(self.convolution_shape[1:])

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