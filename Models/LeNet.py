from NeuralNetwork import NeuralNetwork
from Layers import Conv, Pooling, Flatten, FullyConnected, ReLU, SoftMax,Initializers
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer
from Optimization import Loss


def build():
    optimizer = Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(L2_Regularizer(4e-4))

    net = NeuralNetwork(optimizer, Initializers.Xavier(),
        Initializers.Constant(0.0))

    net.loss_layer = Loss.CrossEntropyLoss()
    #C1
    c1 = Conv.Conv((1, 1), (1, 5, 5), 6)
    net.append_layer(c1)
    net.append_layer(ReLU.ReLU())

    #S2
    s2 = Pooling.Pooling((2, 2), (2, 2))
    net.append_layer(s2)

    #C3
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))
    net.append_layer(ReLU.ReLU())

    #S4
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    net.append_layer(Flatten.Flatten())

    #C5
    net.append_layer(FullyConnected.FullyConnected(16 * 7 *7, 120))
    net.append_layer(ReLU.ReLU())

    #F6
    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())

    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())

    return net