from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dL_dy):
        return dL_dy.reshape(self.X_shape)