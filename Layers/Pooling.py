import numpy as np
from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else:
            self.stride_shape = stride_shape

        if isinstance(pooling_shape, int):
            self.pooling_shape = (pooling_shape, pooling_shape)
        else:
            self.pooling_shape = pooling_shape

        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size, channels, y, x = input_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        y_out = int(np.floor((y - pool_y) / stride_y)) + 1
        x_out = int(np.floor((x - pool_x) / stride_x)) + 1

        output = np.zeros((batch_size, channels, y_out, x_out))
        self.max_indices = np.zeros_like(input_tensor, dtype=bool)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(y_out):
                    for j in range(x_out):
                        y_start = i * stride_y
                        y_end = y_start + pool_y
                        x_start = j * stride_x
                        x_end = x_start + pool_x

                        window = input_tensor[b, c, y_start:y_end, x_start:x_end]
                        max_value = np.max(window)

                        output[b, c, i, j] = max_value

                        max_position = np.unravel_index(
                            np.argmax(window),
                            window.shape
                        )

                        self.max_indices[
                            b,
                            c,
                            y_start + max_position[0],
                            x_start + max_position[1]
                        ] = True

        return output

    def backward(self, error_tensor):
        batch_size, channels, y_out, x_out = error_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        gradient_input = np.zeros_like(self.input_tensor)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(y_out):
                    for j in range(x_out):
                        y_start = i * stride_y
                        y_end = y_start + pool_y
                        x_start = j * stride_x
                        x_end = x_start + pool_x

                        window = self.input_tensor[b, c, y_start:y_end, x_start:x_end]

                        max_position = np.unravel_index(
                            np.argmax(window),
                            window.shape
                        )

                        gradient_input[
                            b,
                            c,
                            y_start + max_position[0],
                            x_start + max_position[1]
                        ] += error_tensor[b, c, i, j]

        return gradient_input