import numpy as np



class NeuralNetwork:

    def __init__(self, n_inputs, topology, activation_funcs, activation_func_last):

        # assert!

        # Initial Checks
        if topology[0] != n_inputs:
            raise ValueError(f"Inconsitent topology: n_inputs ({n_inputs}) must be equal than topology[0] ({topology[0]})")

        if len(toplogy) != len(activation_funcs):
            raise ValueError(f"The number of hidden layers differs form the activation functions")


        self.toplogy = topology
        self.act_funcs = act_funcs
        self.activation_func_last = activation_func_last

        self.num_hidden_layers = len(topology)


        self.build_network()




    def build_network(self):
        """

        """

        network = []

        # First layer
        network.append(__HiddenLayer(self.n_inputs,
                                     self.topology[0],
                                     self.activation_funcs[0])     )

        # Hidden layers
        for l in range(1, num_hidden_layers):
            network.append(__HiddenLayer(self.topology[l-1],
                                         self.topology[l],
                                         self.activation_funcs[l])  )




    def forward(self, X):
        """
            Implements forward propagation from the input vector/matrix X

        """
        A = X
        for layer in network:
            A = layer.activation_func(layer.W @ A + layer.b)

        self.Y_hat = np.copy(A)
        return self.Y_hat




    def train(self):
        """
            Training using backpropagation

        """
        x = 0

        for n in range(0, epochs):
        # TODO




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
        def __linear(self, x)
            return x

        def __grad_linear(self, x)
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
