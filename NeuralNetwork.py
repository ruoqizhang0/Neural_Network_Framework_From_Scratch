import copy
import os
import pickle

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
            self.loss.append(float(loss))
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

    def __getstate__(self):
        state = self.__dict__.copy()

        # remove data layer because generator cannot be pickled
        if "data_layer" in state:
            state["data_layer"] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # initialize removed objects
        self.data_layer = None

    import pickle
    @staticmethod
    def save(filename, net):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(net, f)

    @staticmethod
    def load(filename, data_layer):

        with open(filename, "rb") as f:
            net = pickle.load(f)

        # restore data layer
        net.data_layer = data_layer

        return net