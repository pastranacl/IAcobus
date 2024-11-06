import numpy as np
import dill
from tqdm import tqdm


class NeuralNetwork:
    """




    """


    EPS = 1e-12     # Constant used to prevent numerical issues


    def __init__(self, n_inputs, topology, act_funcs, cost_func):

        # Initial checks
        if len(topology) != len(act_funcs): raise ValueError(f"The number of hidden layers differs form the activation functions")
        if any(af not in ['relu', 'sigmoid', 'tanh', 'linear', 'heavyside', 'softmax'] for af in act_funcs): raise ValueError(f"Unknown activation function")
        if cost_func not in ['mse', 'bin-cross-entropy', 'cross-entropy']: raise ValueError(f"Unknown loss function")


        # Set up the network
        self.n_inputs = n_inputs
        self.topology = topology
        self.act_funcs = act_funcs
        self.num_hidden_layers = len(topology)

        if cost_func == 'mse':
            self.cost = self.__cost_norm
            self.cost_grad = self.__cost_grad_norm
        elif cost_func == 'bin-cross-entropy':
            self.cost = self.__cost_binary_cross_entropy
            self.cost_grad = self.__cost_grad_binary_cross_entropy
        elif cost_func == 'cross-entropy':
            self.cost = self.__cost_cross_entropy
            self.cost_grad = self.__cost_grad_cross_entropy

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

        if X.shape[0] != self.n_inputs:
            print(f"The number of features ({X.shape[0]}) is not equal to the inputs indicated")
            return -1 #exit()

        A_prev = X
        for layer in self.network:
            layer.Z = layer.W @ A_prev + layer.b
            layer.A = layer.activation_func(layer.Z)
            A_prev = layer.A

        self.Y_hat = np.copy(A_prev)
        return self.Y_hat



    def backpropagation(self, X, Y):
        """
            Backpropagation algorithm in vectorised form

            Parameters
            ----------
            X : np.array[inputs values, samples]
                The input vector to be used as input of the network
            Y : np.array[ouput values, samples]
                Associated values to X

        """

        # 1. First, forward propagation
        self.forward(X)

        # 2.1 Backpropagation  (Last layer)
        last_layer = self.network[-1]
        delta = self.cost_grad(Y) * last_layer.grad_activation_func(last_layer.Z)
        last_layer.dJdb = np.sum(delta, axis=1, keepdims=True) # NOTE: LOOK THE SUM HERE...
        last_layer.dJdW = np.sum(delta @ last_layer.A.T, axis=1, keepdims=True) # NOTE: LOOK THE SUM HERE...


        # 2.2 Backpropagation (remaining layers)
        for l in reversed(range(0, self.num_hidden_layers-1)):

            layer = self.network[l]
            next_layer = self.network[l+1]

            A_prev = self.network[l-1].A if l > 0 else X

            delta = (next_layer.W.T @ delta) * layer.grad_activation_func(layer.Z)
            layer.dJdb =   np.sum(delta, axis=1, keepdims=True) # NOTE: LOOK THE SUM HERE...
            layer.dJdW = delta @ A_prev.T



    def train(self, X, Y, num_epochs=5, batch_size=1, learning_rate=1e-3, verbose = True):
        """
            Training of the network using backpropagation

            Parameters
            ----------
            X : ndarray, shape(n_inputs, observations)
                The input vector to be used as input of the network
            Y : ndarray, shape(n_outputs, observations)
                 Associated values or labels to the input features in X
            num_epochs : int, optional
                        Number of passes to the data
            batch_size : int, optional
                        Dimension of the batch, i.e., number of samples to use for
                        training
            learning_rate : float
                            Initial value to consider for training
            verbose : boolean, optional
                      Prints the epoch number and the cost during the training loop

            Returns
            -------
            cost_epoch : ndarray, shape(num_epochs)
                        Evolution of the cost function during the training procedure

        """
        if X.shape[1] != Y.shape[1]:
            print("Non compatible number of observations: X.shape[1] != Y.shape[1]")
            return -1

        if X.shape[0] != self.n_inputs:
            print(f"The number of features ({X.shape[0]}) is not equal to the inputs indicated")
            return -1

        Omega_tot = X.shape[1]
        n_chunks = Omega_tot/batch_size

        X_batches = np.array_split(X, n_chunks, axis=1)
        Y_batches = np.array_split(Y, n_chunks, axis=1)

        cost_epoch = np.zeros(num_epochs)

        for epoch in tqdm(range(0, num_epochs)):
            for x, y in zip(X_batches,  Y_batches):

                self.backpropagation(x,y)

                # Update every layer
                for l, layer in enumerate(self.network):
                    # Momentum

                    # RMSprop

                    # Bias correction

                    # Gradient descent
                    layer.W -= learning_rate * layer.dJdW
                    layer.b -= learning_rate * layer.dJdb

                # Assign the cost to the epoch
                # It corresponds to before the update, since we did not call forward prop yet
                cost_epoch[epoch] += self.cost(y)


            # Average cost for the Omega observations
            cost_epoch[epoch] /= Omega_tot

            # Update progress bar
            if verbose == True:
                tqdm.write(f" Epoch {epoch}/{num_epochs}; Cost: {cost_epoch[epoch]}")


        return cost_epoch





    def __build_network(self):
        """
            Create all the layer and neurons of the network considering the
            specified topolpogy and activation functions.
        """

        self.network = []

        # Create First layer
        self.network.append(self.__HiddenLayer(self.n_inputs,
                                                self.topology[0],
                                                self.act_funcs[0])      )

        # Create Hidden layers
        for l in range(1, self.num_hidden_layers):
            self.network.append(self.__HiddenLayer(self.topology[l-1],
                                                    self.topology[l],
                                                    self.act_funcs[l])   )


    class __HiddenLayer():
        """
            # TODO
            DOCUMENT THE CLASS
        """

        def __init__(self, n_neurons_prev, n_neurons, act_func="relu"):

            # Intialisation of the Weight matrix (He initialisation) and the bias vector
            self.W =  np.random.rand(n_neurons, n_neurons_prev)*np.sqrt(2.0 / n_neurons_prev)
            self.b = np.random.rand(n_neurons, 1)


            # Gradients of the cost with respect to the weigth matrix and bias
            self.dJdW =  np.zeros((n_neurons, n_neurons_prev))
            self.dJdb =  np.zeros((n_neurons, 1))

            # The linar activation Z = WA^{l-1} + b parameter and the Activation A
            # are first created during forward passing. At this stage we do not know
            # the number of samples, i.e., columns in A and Z.

            # Assign activation function
            if act_func == "relu":
                self.activation_func = self.__relu
                self.grad_activation_func = self.__grad_relu

            elif act_func == "sigmoid":
                self.activation_func = self.__sigmoid
                self.grad_activation_func = self.__grad_sigmoid

            elif act_func == "tanh":
                self.activation_func = self.__tanh
                self.grad_activation_func = self.__grad_tanh

            elif act_func == "linear":
                self.activation_func = self.__linear
                self.grad_activation_func = self.__grad_linear

            elif act_func == "heavyside":
                self.activation_func = self.__heavyside
                self.grad_activation_func = self.__grad_heavyside

        # ----------------------------------------------#
        #                Activation functions           #
        #-----------------------------------------------#
        def __linear(self, z):
            return z

        def __grad_linear(self, z):
            return np.ones_like(z)


        def __relu(self, z):
            return np.maximum(0.0, z)

        def  __grad_relu(self, z):
            return  np.where(z > 0., 1.0, 0.0)


        def __sigmoid(self, z):
            return 1/(1 + np.exp(-z))

        def __grad_sigmoid(self, z):
            return self.__sigmoid(z)*(1 - self.__sigmoid(z))


        def __tanh(self, z):
            return np.tanh(X)

        def __grad_tanh(self, z):
            return 1-np.tanh(z)**2


        def __softmax(self, z):
            Q = np.sum(np.exp(z), axis=0, keepdims=True)
            return np.exp(z)/Q

        def __grad_softmax(self, z):
            S = self.__softmax(z)
            return self.__softmax(z) * np.ones()


        def __heavyside(self, z):
            return np.where(z > 0., 1.0, 0.0)

        def __grad_heavyside(self, z):
            return np.where(z == 0., 1.0, 0.0)


    # ----------------------------------------------#
    #                Cost functions                 #
    #-----------------------------------------------#
    def __cost_norm(self, Y):
        dY = Y - self.Y_hat
        #return np.mean(dY*dY)
        return np.sum(dY*dY)/(2*Y.shape[1])

    def __cost_grad_norm(self, Y):
        dY =  self.Y_hat - Y
        #return dY/Y.shape[1]
        return np.sum(dY, keepdims=True)/Y.shape[1]




    def __cost_binary_cross_entropy(self, Y):
        return  np.mean((Y - 1)*np.log(1 - self.Y_hat + NeuralNetwork.EPS) - Y*np.log(self.Y_hat + NeuralNetwork.EPS))
        #return  np.sum((Y - 1)*np.log(1 - self.Y_hat + NeuralNetwork.EPS) - Y*np.log(self.Y_hat + NeuralNetwork.EPS))/Y.shape[1]

    def __cost_grad_binary_cross_entropy(self, Y):
        return  ((1 - Y)/(1 - self.Y_hat + NeuralNetwork.EPS) - Y/self.Y_hat)/Y.shape[1]
        #return  np.sum( (1 - Y)/(1 - self.Y_hat + NeuralNetwork.EPS) - Y/self.Y_hat, keepdims=True )



    def __cost_cross_entropy(self, Y):
        return -np.sum(Y*np.log(self.Y_hat + NeuralNetwork.EPS))/Y.shape[1]

    def __cost_grad_cross_entropy(self, Y):
        return -(Y / self.Y_hat)/Y.shape[1]

    """
        Accesory functionalities
    """


    def one_hot_encoding(self, Y, n_classes):
        """
            Creates one hot encoded reppresentations of an input
            vector indicating the class on each element

            Parameters
            -------
            Y : ndarray, shape(observations)
                Array of labels
            n_classes : integer
                        Total number of possible classes

            Returns
            -------
            one_hot_Y : ndarray, shape(n_classes, observations)
                        One-hot encoded reppresentations of Y

        """
        if np.min(Y)<0 or np.max(Y) > n_classes -1:
            print("Some ids of the classes are not valid!")
            return -1

        Omega = len(Y)

        one_hot_Y = np.zeros((n_classes, Omega))
        one_hot_Y[Y, np.arange(0,Omega)] = 1

        return one_hot_Y


    def split_data_train_test(self, X, Y, p_train=0.6):
        """
            Splits the data into training and test sets

            Parameters
            ----------
            X : ndarray, shape(n_inputs, observations)
                Array of features
            Y : ndarray, shape(n_outputs, observations)
                Array of labels
            p_train : float, optional
                      Relative proportion of the total data to use
                      for the training set

            Returns
            -------
            X_train : ndarray, shape(n_inputs, observations*p_train)
                      Array of features for training
            Y_train : ndarray, shape(n_ouputs, observations*p_train)
                      Array of labels for training
            X_test : ndarray, shape(n_inputs, observations*(1-p_train))
                      Array of features for test set
            Y_test : ndarray, shape(n_ouputs, observations*(1-p_train))
                      Array of labels for test set
        """

        Omega = X.shape[1]

        indices = np.arange(0, Omega-1)
        idx_train = np.int32(np.random.choice(indices,
                                            int(np.round(p_train*Omega)),
                                            replace=False) )

        mask = np.zeros(Omega, dtype=bool)
        mask[idx_train] = 1

        X_train = X[:, mask ]
        Y_train = Y[:, mask ]
        X_test = X[:, ~mask ]
        Y_test = Y[:, ~mask ]

        return X_train, Y_train, X_test, Y_test


    def export(self, file_name):
        """
            Save a pickled copy of the class

            Parameters
            ----------
            file_name : string
                        Path and filename of the pickle pkl file to be saved
        """
        with open(file_name, 'wb') as f:
            dill.dump(self, f)


    @classmethod
    def load(cls, file_name):
        """
            Load (unpickle) a pickled copy of the class
            (the classmethod decorator allows its usage without
            an instance, i.e., a precreated object

            Parameters
            ----------
            file_name : string
                        Path and filename of the .pkl file to load

        """
        #ryc = neurnet.load('my_instance.pkl')  # Load instance
        with open(file_name, 'rb') as f:
            return dill.load(f)
