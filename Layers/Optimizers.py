class Sgd:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.alpha * gradient_tensor