import numpy as np
from Optimization.Constraints import L1_Regularizer


class Optimizer:
    def __init__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.alpha * gradient_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, alpha, momentum):
        super().__init__()
        self.alpha = alpha
        self.momentum = momentum
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.alpha* self.regularizer.calculate_gradient(weight_tensor)
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum * self.velocity - self.alpha * gradient_tensor
        weight_tensor += self.velocity
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1, beta2):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.update = None
        self.velocity = None
        self.epsilon = np.finfo(float).eps
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - \
                            self.learning_rate * \
                            self.regularizer.calculate_gradient(weight_tensor)

        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
            self.update = np.zeros_like(weight_tensor)

        self.k += 1

        self.velocity = self.beta1 * self.velocity + (1 - self.beta1) * gradient_tensor
        self.update = self.beta2 * self.update + (1 - self.beta2) * gradient_tensor * gradient_tensor

        hat_velocity = self.velocity / (1 - self.beta1 ** self.k)
        hat_update = self.update / (1 - self.beta2 ** self.k)

        weight_tensor -= (self.learning_rate * hat_velocity) / (np.sqrt(hat_update) + self.epsilon)

        return weight_tensor