import numpy as np

class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant)

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(low = 0.0, high = 1.0, size = weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(loc = 0.0, scale = np.sqrt(2.0 / (fan_in + fan_out)), size = weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(loc = 0.0, scale = np.sqrt(2. / fan_in), size = weights_shape)