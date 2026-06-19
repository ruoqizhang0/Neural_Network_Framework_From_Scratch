import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

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
            dL_dy = layer.backward(dL_dy)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(
                self.weights_initializer,
                self.bias_initializer
            )

        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False

        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, x):
        self.phase = True

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def norm(self, weights):
        for layer in self.layers:
            layer.norm()
            

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

        for layer in self.layers:
            layer.testing_phase = value