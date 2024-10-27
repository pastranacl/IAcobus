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

        self.__build_network()



    def forward(self, X):
        """
            Implements forward propagation from the input vector/matrix X
            The first time that is called, this function creates the linear Z and the
            non-linear actvation A for each layer of the network

            Parameters
            ----------
            X : np.array[n_inputs, samples]
                The input vector to be used as input of the network

            Returns
            -------

            Y_hat :  np.array[n_outputs, samples]
                    Prediction given by the network

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

            Parameters
            ----------
            X : np.array[inputs values, samples]
                The input vector to be used as input of the network
            Y : np.array[inputs values, samples]
                Associated values to X

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



    def train(self, X, Y, num_epochs=5, batch_size=1):
        """
            Training of the network using backpropagation

            Parameters
            ----------
            X : ndarray, shape(n_inputs, observations)
                DEFINITION TODO
            Y : ndarray, shape(n_outputs, observations)
                DEFINITION TODO

            num_epochs : int, optional
                        DEFINITION TODO

            batch_size : int, optional
                        DEFINITION TODO

            Returns
            -------
            cost_epoch : ndarray, shape(num_epochs)
                        Evolution of the cost function during the training procedure

        """
        assert X.shape[1] != Y.shape[1], "Non compatible number of observations: X.shape[1] != Y.shape[1]"

        Omega_tot = X.shape[1]
        n_chunks = Omega_tot/batch_size

        X_batches = np.array_split(X, n_chunks, axis=1)
        Y_batches = np.array_split(Y, n_chunks, axis=1)


        cost_epoch = np.zeros(num_epochs)

        for epoch in tqdm(range(1, num_epochs)):

            for x, y in zip(X_batches,  Y_batches):

                self.backpropagation(x,y)

                # Update every layer
                for layer in self.layers:

                    # Momentum

                    # RMSprop

                    # Bias correction

                    # Gradient descent
                    layer.W -= 0.1*layer.dJdW
                    layer.b -= 0.1*layer.dJdb


                # Assign the cost to the epoch (which corresponds to before the update)
                cost_epoch[epoch] = self.cost(y)


            # Average cost
            cost_epoch[epoch] /= Omega_tot

            # Update progress bar
            tqdm.write(f"Epoch {epoch}/{num_epochs}; Cost: {cost_epoch[epoch]}")


        return cost_epoch





    def __build_network(self):
        """
            Create all the neurons of the network.
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


    class __HiddenLayer:

        """
            # TODO
            DOCUMENT THE CLASS
        """

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



        # ----------------------------------------------#
        #                Activation functions           #
        #-----------------------------------------------#
        def __linear(self, z):
            return z

        def __grad_linear(self, z):
            return np.ones(z.size)


        def __relu(self, z):
            return np.maximum(0, z)

        def  __grad_relu(self, z):
            return  1 if z > 0 else 0


        def __sigmoid(self, z):
            return 1/(1 + exp(-z))

        def __grad_sigmoid(self, z):
            return __sigmoid(z)*(1 - __sigmoid(z))


        # ----------------------------------------------#
        #                Cost functions                 #
        #-----------------------------------------------#
        def __cost_norm(Y):
            dY = self.Y_hat - Y
            return np.sum( np.dot(dY,dY) )/Y.shape[1]


        def __cost_grad_norm(Y):
            dY =  self.Y_hat - Y
            return np.sum(dY)/Y.shape[1]





if __name__ == '__main__':

    # Add test!
    print("Hello World!")
