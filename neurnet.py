import numpy as np



class NeuralNetwork:

    def __init__(self, n_inputs, topology, activation_funcs, activation_func_last):


        # Initial Checks
        if topology[0] != n_inputs:
            raise ValueError(f"Inconsitent topolgy: n_inputs ({n_inputs}) must be equal than topology[0] ({topology[0]})")



        self.toplogy = topology
        self.act_funcs = act_funcs
        self.activation_func_last = activation_func_last

        self.num_hidden_layers = len(topology)






    def build_network(self):
        """

        """

        network = []

        for l in range(0, num_hidden_layers):
            network.append()


    def train(self):
        x = 0




    class __HiddenLayer:

        def __init__(self, n_neurons_prev, n_neurons, act_func="relu"):

            self.W = np.random.rand(n_neurons, n_neurons)
            self.b = np.random.rand(n_neurons, 1)

            if act_func == "relu":
                self.activation_func = self._relu()
                self.grad_activation_func = self._grad_relu()


        def __relu(self, x):
            return np.maximum(0, x)

        def  __grad_relu(self, x):
            return  1 if x > 0 else 0
