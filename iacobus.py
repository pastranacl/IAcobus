"""
    IAcobus

    Author: Cesar L. Pastrana
    Date: 2024
    License: GNU General Public License v3.0 (GPL-3.0)

    Description:
    Library implementing a neural network at CPU level relying solely on Numpy

    Copyright (c) 2024, Cesar L.Pastrana

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np
import dill
from tqdm import tqdm


class IAcobus:
    """

        IAcobus self-containd class to build, train and preprocess fully-conected neural networks

        Attributes
        ----------
            n_inputs : int
                Number of neurons of the inpput layer = numer of features
            topolgy : array of int
                Number of neurons on each layer
            act_func  :  array of strings
                Activation function to use in each layer
            cost_fuct : string
                Cost function to use

        Methods (all private)
        -------


    """


    EPS = 1e-12     # Constant used to prevent numerical issues


    def __init__(self, n_inputs, topology, act_funcs, cost_func):

        # Initial checks
        if len(topology) != len(act_funcs): raise ValueError(f"The number of hidden layers differs form the activation functions")
        if any(af not in ['relu', 'sigmoid', 'tanh', 'linear', 'heavyside', 'softmax'] for af in act_funcs): raise ValueError(f"Unknown activation function")
        if cost_func not in ['mse', 'bin-cross-entropy', 'cross-entropy']: raise ValueError(f"Unknown loss function")

        # Checks for the softmax
        for l, af in enumerate(act_funcs):
            if af=='softmax' and l != len(act_funcs) - 1:
                raise ValueError(f"In the current implementatiom Softmax layer is only possible on the last layer")

        if act_funcs[-1] == 'softmax' and cost_func != 'cross-entropy':
            raise ValueError(f"In the current implementatiom Softmax layer can only be combined with cross-entropy")


        # Set up the network
        self.n_inputs = n_inputs
        self.topology = topology
        self.act_funcs = act_funcs
        self.num_hidden_layers = len(topology)
        self.cost_func = cost_func

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
        next_to_last_layer = self.network[-2]
        #delta = self.cost_grad(Y)[:,np.newaxis,:] * last_layer.grad_activation_func(last_layer.Z)

        if last_layer.act_func=='softmax' and self.cost_func=='cross-entropy':
            delta = last_layer.activation_func(last_layer.Z) - Y
        else:
            delta = self.cost_grad(Y) * last_layer.grad_activation_func(last_layer.Z)


        last_layer.dJdb = np.sum(delta, axis=1, keepdims=True) # NOTE: LOOK THE SUM HERE...
        last_layer.dJdW = delta @ next_to_last_layer.A.T


        # 2.2 Backpropagation (remaining layers)
        for l in reversed(range(0, self.num_hidden_layers-1)):

            layer = self.network[l]
            next_layer = self.network[l+1]

            A_prev = self.network[l-1].A if l > 0 else X

            delta = (next_layer.W.T @ delta) * layer.grad_activation_func(layer.Z)
            layer.dJdb =   np.sum(delta, axis=1, keepdims=True) # NOTE: LOOK THE SUM HERE...
            layer.dJdW =   delta @ A_prev.T



    def __gradient_descendent(self, learning_rate=1e-3, *args, **kwargs):
        """
            Basic gradient descendent algorithm to update network parameters
        """
        def gd_update(*args, **kwargs):
            for l, layer in enumerate(self.network):
                layer.W -= learning_rate * layer.dJdW
                layer.b -= learning_rate * layer.dJdb

        return gd_update


    def __gradient_descendent_momentum(self, learning_rate=1e-3, beta=0.9):
        """
            Python closure function that implement gradient descendent with momentum
            and no bias correction.

        """
        # Initialise the first  arrays/vector for the weighs and biases
        W_ms = []
        b_ms = []

        for l, layer in enumerate(self.network):
            W_ms.append(np.zeros_like(layer.dJdW))
            b_ms.append(np.zeros_like(layer.dJdb))


        def gdm_update(*args, **kwarg):
            """
                Update rule for gradient descendent with momentum
            """
            nonlocal W_ms, b_ms
            for l, layer in enumerate(self.network):
                # First moment estimate
                W_ms[l] = beta*W_ms[l] + (1-beta)*layer.dJdW
                b_ms[l] = beta*b_ms[l] + (1-beta)*layer.dJdb

                # Update Parameters
                layer.W -= learning_rate*W_ms[l]
                layer.b -= learning_rate*b_ms[l]

        return gdm_update



    def __adam(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8):
        """
            Python closure function that implement the Adam optimiser

            Parameters
            ----------
                learning_rate : float
                                Basal lerning rate, alpha in the original paper
                beta1 : float
                        Exponential decay rates for the first moment estimate
                beta2 : float
                        Exponential decay rates for the second moment estimate

            Returns
            -------
                adam_update : function
                              Update rule to be used in the training loop

        """

        # Initialise the first and second moment arrays/vector for the weighs and biases
        W_ms = []
        b_ms = []
        W_vs = []
        b_vs = []

        for l, layer in enumerate(self.network):

            W_ms.append(np.zeros_like(layer.dJdW))
            b_ms.append(np.zeros_like(layer.dJdb))

            W_vs.append(np.zeros_like(layer.dJdW))
            b_vs.append(np.zeros_like(layer.dJdb))


        def adam_update(t):
            """
                Implements the Adam algorithm with bias correction as
                described in ADAM: A Method for Stochastic Optimization
                by Kigma D.P and Ba J.L. ICLR (2015)

                Parameters
                -----------
                t   : int.
                      Epoch number for the bias correction step
            """

            # Keep state persistent across calls
            nonlocal W_ms, b_ms, W_vs, b_vs
            for l, layer in enumerate(self.network):


                # First moment estimate
                W_ms[l] = beta1*W_ms[l] + (1-beta1)*layer.dJdW
                b_ms[l] = beta1*b_ms[l] + (1-beta1)*layer.dJdb

                # Second raw moment estimates
                W_vs[l] = beta2*W_vs[l] + (1-beta2)*layer.dJdW*layer.dJdW
                b_vs[l] = beta2*b_vs[l] + (1-beta2)*layer.dJdb*layer.dJdb

                # Bias corrections
                dJdW_m_hat =  W_ms[l]/(1-beta1**(t+1))
                dJdb_m_hat =  b_ms[l]/(1-beta1**(t+1))

                dJdW_v_hat =  W_vs[l]/(1-beta2**(t+1))
                dJdb_v_hat =  b_vs[l]/(1-beta2**(t+1))

                # Update Parameters
                layer.W -= learning_rate*dJdW_m_hat/(np.sqrt(dJdW_v_hat) + eps_adam)
                layer.b -= learning_rate*dJdb_m_hat/(np.sqrt(dJdb_v_hat) + eps_adam)


        return adam_update



    def __adopt(self, X, Y, learning_rate=1e-3, beta1=0.9, beta2=0.9999, eps_adopt=1e-6):
        """
            Python closure function that implement the ADOPT optimiser

            Parameters
            ----------
                learning_rate : float
                                Basal lerning rate, alpha in the original paper
                beta1 : float
                        Exponential decay rates for the first moment estimate
                beta2 : float
                        Exponential decay rates for the second moment estimate

            Returns
            -------
                adop_update : function
                              Update rule to be used in the training loop

        """

        # Initialise the first and second moment arrays/vector for the weighs and biases
        self.backpropagation(X,Y)

        W_vs = []
        b_vs = []
        W_ms = []
        b_ms = []

        for l, layer in enumerate(self.network):

            W_vs.append(layer.dJdW*layer.dJdW)
            b_vs.append(layer.dJdb*layer.dJdb)

            W_ms.append(layer.dJdW / np.maximum(np.sqrt(layer.dJdW*layer.dJdW), eps_adopt) )
            b_ms.append(layer.dJdb / np.maximum(np.sqrt(layer.dJdb*layer.dJdb), eps_adopt) )


        def adopt_update(t):
            """
                Implements the Adopt algorithm as described in:
                ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate by Taniguchi S. et al.
                https://arxiv.org/pdf/2411.02853 (2024)

                Parameters
                -----------
                t   : int.
                    Epoch number for the bias correction step
            """


            nonlocal W_ms, b_ms, W_vs, b_vs  # Keep state persistent across calls
            for l, layer in enumerate(self.network):

                W_ms[l]  = beta1*W_ms[l] + (1-beta1)*layer.dJdW/np.maximum(np.sqrt(W_vs[l]), eps_adopt)
                layer.W -= learning_rate*W_ms[l]
                W_vs[l]  = beta2*W_vs[l] + (1-beta2)*layer.dJdW*layer.dJdW

                b_ms[l]  = beta1*b_ms[l] + (1-beta1)*layer.dJdb/np.maximum(np.sqrt(b_vs[l]), eps_adopt)
                layer.b -= learning_rate*b_ms[l]
                b_vs[l]  = beta2*b_vs[l] + (1-beta2)*layer.dJdb*layer.dJdb


        return adopt_update



    def train(self, X, Y, num_epochs=5, batch_size=1, learning_rate=1e-3, algorithm='adam', verbose = True, **kwargs):
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

        if algorithm not in ['gd', 'gdm', 'adam', 'adopt']:
            print("The specified optimisation algorithm is not valid.")
            return -1


        # Selection of minimisation algorithm
        if algorithm == 'gd':
            minimiser = self.__gradient_descendent(learning_rate=learning_rate)
        elif algorithm == 'gdm':
            minimiser = self.__gradient_descendent_momentum(learning_rate=learning_rate, **kwargs)
        elif algorithm == 'adam':
            minimiser = self.__adam(learning_rate=learning_rate, **kwargs)
        elif algorithm == 'adopt':
            X_batches,  Y_batches = self.__gen_batches(X, Y, batch_size)
            minimiser = self.__adopt(X_batches[0], Y_batches[0], learning_rate=learning_rate, **kwargs)


        # Main training loop
        Omega_tot = X.shape[1]
        cost_epoch = np.zeros(num_epochs)

        pbar = tqdm(total=num_epochs)
        for epoch in range(0, num_epochs):

            # Generate and iterate over batches
            X_batches,  Y_batches = self.__gen_batches(X, Y, batch_size)
            for x, y in zip(X_batches,  Y_batches):
                self.backpropagation(x,y)
                cost_epoch[epoch] += self.cost(y)*x.shape[1]    # Weighted sum
                minimiser(epoch)

            # Average cost for the Omega observations
            cost_epoch[epoch] /=  Omega_tot

            # Update progress bar
            if epoch % 10 == 0 and verbose == True:
                pbar.set_description_str(f"Cost: {cost_epoch[epoch]:.5g};   Epoch: {epoch}/{num_epochs}")


            pbar.update(1)

        return cost_epoch


    def __gen_batches(self, X, Y, batch_size):
        """
            Generate batches, shuffling and spliting the full data set

            Parameters
            ----------

            X : ndarray, shape(n_inputs, observations)
                Array of input features

            Y : ndarray, shape(n_inputs, observations)
                Array of output/labels

            batch_size : int
                        Number of observations per batch
            Returns
            -------
            X_batches : ndarray, shape(n_inputs, observations, batch_size)
                        Array of batched data with the features for training

            Y_batches : ndarray, shape(n_inputs, observations, batch_size)
                        Array of batched data outputs for training

        """
        Omega_tot = X.shape[1]
        n_chunks = Omega_tot/batch_size
        indices = np.arange(Omega_tot)
        np.random.shuffle(indices)

        X_batches = np.array_split(X[:,indices], n_chunks, axis=1)
        Y_batches = np.array_split(Y[:,indices], n_chunks, axis=1)

        return X_batches, Y_batches



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
        __HiddenLayer is private class that creates a layer of neurons, attenting to
        the number of neurons of the previous layer. The the weigths


        Attributes
        ----------
        n_neurons_prev : int
            Number of neurons of the previous layer
        n_neurons : int
            Number of neurons of this layer
        act_func  :  string
            Activation function to use in all the neurons of the layer

        Methods (all private)
        -------
        __linear()
            Linear activation function.
        __grad_linear()
            Gradiento of the linear activation function
        __sigmoid()
            Sigmoid activation function.
        __grad_sigmoid()
            Gradient of the sigmoid activation function
        __tanh()
            Hyperbolic tangent activation function.
        __grad_tanh()
            Gradient of the hyperbolic tanget activation function
            __heavyside()
            Heavyside activation function.
        __grad_heavyside
            Gradient of the Heavyside activation function

        """

        def __init__(self, n_neurons_prev, n_neurons, act_func="relu"):
            """
                Initialised the hidden layer
            """


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
            self.act_func = act_func
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

            elif act_func == "softmax":
                self.activation_func = self.__softmax
                self.grad_activation_func = self.__grad_softmax


        #   Activation functions
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

            S_omega =  S[:, :, np.newaxis]
            diag= S_omega * np.eye((S.shape[0]))[:,np.newaxis,:]

            outer =  S_omega * S_omega.transpose(2,1,0)
            gr = diag - outer
            return gr.transpose(0,2,1)


        def __heavyside(self, z):
            return np.where(z > 0., 1.0, 0.0)

        def __grad_heavyside(self, z):
            return np.where(z == 0., 1.0, 0.0)


    #   Cost functions
    def __cost_norm(self, Y):
        dY = Y - self.Y_hat
        return np.sum(dY*dY)/(2*Y.shape[1])

    def __cost_grad_norm(self, Y):
        dY =  self.Y_hat - Y
        return np.sum(dY, axis=1, keepdims=True)/Y.shape[1]



    def __cost_binary_cross_entropy(self, Y):
        return  np.mean((Y - 1)*np.log(1 - self.Y_hat + self.EPS) - Y*np.log(self.Y_hat + self.EPS))

    def __cost_grad_binary_cross_entropy(self, Y):
        return  np.sum(((1 - Y)/(1 - self.Y_hat + self.EPS) - Y/self.Y_hat),  axis=1, keepdims=True )/Y.shape[1]



    def __cost_cross_entropy(self, Y):
        return -np.sum(Y*np.log(self.Y_hat + self.EPS))/Y.shape[1]

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
        with open(file_name, 'rb') as f:
            return dill.load(f)
