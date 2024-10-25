import numpy as np
from tqdm import tqdm


class NeuralNetwork:
    """




    """

    def __init__(self, n_inputs, topology, activation_funcs, cost_function):

        # assert!

        # Initial Checks
        if topology[0] != n_inputs:
            raise ValueError(f"Inconsitent topology: n_inputs ({n_inputs}) must be equal than topology[0] ({topology[0]})")

        if len(toplogy) != len(activation_funcs):
            raise ValueError(f"The number of hidden layers differs form the activation functions")


        self.toplogy = topology
        self.act_funcs = act_funcs
        self.num_hidden_layers = len(topology)

        if cost_funct=='norm':
            self.cost = __norm()
            self.cost_grad = __grad_norm()

        self.build_network()



    def build_network(self):
        """

        """

        self.network = []

        # Create First layer
        self.network.append(__HiddenLayer(self.n_inputs,
                                     self.topology[0],
                                     self.activation_funcs[0])      )

        # Create Hidden layers
        for l in range(1, num_hidden_layers):
            self.network.append(__HiddenLayer(self.topology[l-1],
                                         self.topology[l],
                                         self.activation_funcs[l])   )




    def forward(self, X):
        """
            Implements forward propagation from the input vector/matrix X
            The first time that is called, this function creates the linear Z and the
            non-linear actvation A for each layer of the network

            Input:   X np.array[inputs values, samples] = The input vector to be used as input of the network

        """
        A = X
        for layer in self.network:
            layer.Z = layer.W @ A + layer.b
            layer.A = layer.activation_func(layer.Z)
            A = np.copy(layer.A)

        self.Y_hat = np.copy(A)
        return self.Y_hat



    def backpropagation(self, X, Y):
        """
            Backpropagation algorithm in vectorised form

            Input:  X np.array[inputs values, samples] = The input vector to be used as input of the network
                    Y np.array[inputs values, samples] = Associated values to X

        """

        # First, forward propagation
        self.forward(X)

        # Last layer
        L = self.n_neurons-1
        last_layer = self.network[L]
        delta = self.grad_cost(Y, last_layer.A)*last_layer.grad_activation_func(last_layer.Z)

        last_layer.dJdb = np.sum(delta, axis=1, keepdims=True)
        last_layer.dJdW = delta @ last_layer.A.T

        # Remaining layers
        for l in reversed(range(0, self.n_neurons-1)):

            layer = self.network[l]
            next_layer = self.network[l+1]

            delta = (next_layer.T @ delta) * layer.grad_activation_func(layer.Z)
            layer.dJdb =  np.sum(delta, axis=1, keepdims=True)
            layer.dJdW = delta @ layer.A.T



    def train(self, X, Y, epochs=5):
        """
            Training using backpropagation

        """

        for n in tqdm(range(0, epochs)):

            # TODO
            self.backpropagation(X, Y)

            for layer in self.layers:

                # Momentum

                # RMSprop

                # Bias correction

                # Gradient descent
                layer.W -= 0.1*layer.dJdW
                layer.b -= 0.1*layer.dJdb



    def __cost_norm(Y):
        dY = self.Y_hat - Y
        return np.sum( np.dot(dY,dY) )/Y.shape[1]


    def __cost_grad_norm(Y):
        dY =  self.Y_hat - Y
        return np.sum(dY)/Y.shape[1]



    class __HiddenLayer:

        def __init__(self, n_neurons_prev, n_neurons, act_func="relu"):

            # Weight matrix and bias
            self.W = np.random.rand(n_neurons, n_neurons_prev)
            self.b = np.zeros((n_neurons, 1))


            # Gradients of the cost with respect to the weigth matrix and bias
            self.dJdW =  np.random.rand(n_neurons, n_neurons_prev)
            self.dJdb =  np.zeros((n_neurons, 1))


            # Assign activation function
            if act_func == "relu":
                self.activation_func = self._relu()
                self.grad_activation_func = self._grad_relu()

            elif act_func == "sigmoid":
                self.activation_func = self._sigmoid()
                self.grad_activation_func = self._grad_sigmoid()

            elif act_func == "linear":
                self.activation_func = self._linear()
                self.grad_activation_func = self._grad_linear()



        """
            Activation functions
        """
        def __linear(self, x):
            return x

        def __grad_linear(self, x):
            return x#TODO

        def __relu(self, x):
            return np.maximum(0, x)

        def  __grad_relu(self, x):
            return  1 if x > 0 else 0

        def __sigmoid(self, x):
            return #TODO

        def __grad_sigmoid(self, x):
            #TODO
            return





if __name__ == '__main__':

    # Add test!
    print("Hello World!")
