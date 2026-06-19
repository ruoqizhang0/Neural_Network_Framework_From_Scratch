import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape

        if isinstance(pooling_shape, int):
            self.pooling_shape = (pooling_shape,)
        else:
            self.pooling_shape = pooling_shape

    def forward(self, X):
        self.X = X
        batches, channels, y, x = X.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        y_out = int(np.floor((y - pool_y) / stride_y)) + 1
        x_out = int(np.floor((x - pool_x) / stride_x)) + 1

        Y = np.zeros((batches, channels, y_out, x_out))

        self.max_indices = np.zeros_like(X, dtype=bool)

        for b in range(batches):
            for c in range(channels):
                for j in range(y_out):
                    for i in range(x_out):
                        y_start = j * stride_y
                        y_end = y_start + pool_y
                        x_start = i * stride_x
                        x_end = x_start + pool_x
                        window = X[b, c, y_start:y_end, x_start:x_end] #create a pool window

                        Y[b, c, j, i] = np.max(window)

                        max_position = np.unravel_index(np.argmax(window), window.shape)

                        self.max_indices[
                            b,
                            c,
                            y_start + max_position[0],
                            x_start + max_position[1]
                        ] = True
        return Y

    def backward(self, dL_dY):
        batches, channels, y, x = dL_dY.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        dL_dX = np.zeros_like(self.X)

        for b in range(batches):
            for c in range(channels):
                for j in range(y):
                    for i in range(x):
                        y_start = j * stride_y
                        y_end = y_start + pool_y
                        x_start = i * stride_x
                        x_end = x_start + pool_x
                        window = self.X[b, c, y_start:y_end, x_start:x_end]

                        max_position = np.unravel_index(
                            np.argmax(window),
                            window.shape
                        )

                        dL_dX[
                            b,
                            c,
                            y_start + max_position[0],
                            x_start + max_position[1]
                        ] += dL_dY[b, c, j, i]

        return dL_dX