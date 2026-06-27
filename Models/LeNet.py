from NeuralNetwork import NeuralNetwork
from Layers import Conv, Pooling, Flatten, FullyConnected, ReLU, SoftMax
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer


def build():
    net = NeuralNetwork()

    optimizer = Adam(5e-4)
    optimizer.add_regularizer(L2_Regularizer(4e-4))
    net.optimizer = optimizer

    net.append_layer(Conv.Conv((1, 5, 5), 6, 1))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    net.append_layer(Conv.Conv((6, 5, 5), 16, 1))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    net.append_layer(Flatten.Flatten())

    net.append_layer(FullyConnected.FullyConnected(16 * 7 * 7, 120))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())

    for layer in net.layers:
        if hasattr(layer, "initialize"):
            layer.initialize(initializer, initializer)

    return net