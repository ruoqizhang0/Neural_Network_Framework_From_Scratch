import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.loss_layer = None
        self.data_layer = None

    def forward(self):
        x, self.y = self.data_layer.next()
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_layer.forward(x, self.y)
        return loss

    def backward(self):
        dL_dy = self.loss_layer.backward(self.y)
        for layer in reversed(self.layers):
            dL_dy= layer.backward(dL_dy)

    def append_layer(self, layer):
        if layer.trainable:
                layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x