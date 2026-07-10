import copy
import numpy as np

from Layers import Base


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        super().__init__()

        self.trainable = True

        self.H = np.zeros(hidden_size)

        self._memorize = False

        self.weights_xh = np.random.rand(
            input_size + hidden_size + 1,
            hidden_size
        )

        self.weights_hy = np.random.rand(
            hidden_size + 1,
            output_size
        )

        self._gradient_weights = None

        self._optimizer = None
        self._optimizer_weights = None

        self.X = None
        self.hidden_states = None
        self.outputs = None

        self.regularizer = None


    @property
    def memorize(self):
        return self._memorize


    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self.weights_xh

    @weights.setter
    def weights(self, value):

        if value is None:
            return

        self.weights_xh = value.copy()

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(optimizer)
        self.regularizer = optimizer.regularizer
    def forward(self, input_tensor):

        batch_size = input_tensor.shape[0]

        # reset memory if sequences are independent
        if not self.memorize:
            self.H = np.zeros(self.hidden_size)


        self.X = input_tensor

        self.hidden_states = []
        self.outputs = []


        H_previous = self.H.copy()


        for t in range(batch_size):

            x = input_tensor[t]


            # concatenate bias + input + previous hidden
            z = np.concatenate(
                ([1], x, H_previous)
            )


            H_current = np.tanh(
                np.dot(z, self.weights_xh)
            )


            y_input = np.concatenate(
                ([1], H_current)
            )

            Y = np.dot(
                y_input,
                self.weights_hy
            )


            self.hidden_states.append(H_current)
            self.outputs.append(Y)


            H_previous = H_current


        # store final hidden state for next sequence
        self.H = H_previous.copy()


        return np.array(self.outputs)



    def backward(self, error_tensor):

        batch_size = self.X.shape[0]


        grad_xh = np.zeros_like(self.weights_xh)
        grad_hy = np.zeros_like(self.weights_hy)

        dX = np.zeros_like(self.X)


        dh_next = np.zeros(self.hidden_size)


        for t in reversed(range(batch_size)):

            H = self.hidden_states[t]


            # output gradient
            hy_input = np.concatenate(
                ([1], H)
            )

            grad_hy += np.outer(
                hy_input,
                error_tensor[t]
            )


            dh = np.dot(
                error_tensor[t],
                self.weights_hy[1:].T
            )

            dh += dh_next


            # tanh derivative
            dtanh = dh * (1 - H ** 2)


            if t == 0:
                H_prev = np.zeros(self.hidden_size)
            else:
                H_prev = self.hidden_states[t-1]


            rnn_input = np.concatenate(
                ([1], self.X[t], H_prev)
            )


            grad_xh += np.outer(
                rnn_input,
                dtanh
            )


            dx = np.dot(
                dtanh,
                self.weights_xh[1:self.input_size+1].T
            )

            dX[t] = dx


            dh_next = np.dot(
                dtanh,
                self.weights_xh[self.input_size+1:].T
            )


        self._gradient_weights = grad_xh


        if self._optimizer is not None:
            new_weights = self._optimizer_weights.calculate_update(
                self.weights,
                self._gradient_weights
            )

            self.weights = new_weights


        return dX



    def initialize(self, weights_initializer, bias_initializer):

        self.weights_xh = weights_initializer.initialize(
            self.weights_xh.shape,
            self.input_size + self.hidden_size,
            self.hidden_size
        )


        self.weights_hy = weights_initializer.initialize(
            self.weights_hy.shape,
            self.hidden_size,
            self.output_size
        )

    def calculate_regularization_loss(self):

        if self.regularizer is not None:
            return (
                    self.regularizer.norm(self.weights_xh)
                    +
                    self.regularizer.norm(self.weights_hy)
            )

        return 0