# Neural Network from Scratch

This is an exercise to build a full neural network from scratch. It use Numpy vectorisation for an optimal performance.
The main code is contained in the neurnet class.


## Instructions
The implementation approach is inspired by the paradigm of TensorFlow + Keras.



1. **Create neural network object:** The neural network is created by specifiying the number of inputs, the topology,  the activation functions for each layer, and hte cost function. The topology of the network as well as the associated activation functions for each layer are provided as arrays. The following code creates the network `ryc`, composed of three hidden layers and one input variable. The two first layers have 64 $\times$ ReLU activation functions, and the last one is a linear function. The cost function is the mean-squared error.

```
    from neurnet import NeuralNetwork

    n_inputs = 1
    costf = 'mse'
    topology = [64, 64, 1]
    activation_funcs = ['relu',
                        'relu',
                        'linear']

    ryc = NeuralNetwork(n_inputs,
                        topology,
                        activation_funcs,
                        cost_func = costf)
```

| Argument        | Description                |
|   ---            | ---                        |
| `n_inputs`          | int. Number of features |
| `topology` | Array containing the number of neurons on each layer |
| `activation_funcs` | Array of strings for the activation function of each layer. The dimensions must coincide with that of `topology`. The activation functions available are the Heavyside `heavyside`, the rectifying linear unit ReLU `relu`, the sigmoid `sigmoid`, and the softmax `softmax`   |
| `cost_func`| String indicating the cost function to use. The options are the mean-squared error `mse`, and binary cross entropy `bin-cross-ent` |

1. **Preprocessing of data:** Traning and validation, description of sizes One hot encoding

3. **Training:** Initially, the weights $W$ are initialised with random numbers (using He-initilisation). Biases $b$ are initilised to zero. Labeled data can be used to train the network, such that the optimal $W$ and $b$ for each neuron of the network is determined as those minimising the cost function. The following code uses the matrix $X$ as input and $Y$ labeled data.
INDICATE HOW YOU SHOULD PROVIDE THE VALUES

```
    cost = ryc.train(X_train, Y_train,
                    num_epochs=5000,
                    batch_size=10,
                    learning_rate=1e-6,
                    verbose=True)
```

The `cost` is an array with the cost for each epoch, useful for reppresenting training curves.

| Argument        | Description                |
|   ---            | ---                        |
| `num_epochs`     | int. The number of optimisation steps, considering as optimisation step the full pass to the data set |
| `batch_size`      | int. Samples to use. For a total number of $\Omega$ samples, and batch size $\omega$, the total number of batches is simply $\Omega/\omega$. Since this library works on the CPU, fitting the data on memory is not a problem, and then this hyperparameter is not so critical. |
| `learning_rate` | float. Initial learning rate for the optimiser |
| `verbose` | bool. Prints the epoch number and the cost during training. |


4. **Forward propagation:** To obtain predictions form an input array `X_test` the `forward` function is called:

```
    Y_pred = ryc.forward(X_test)
```

If test or cross-validation data are available, the associated cost can be calculated using the `cost function:`
```
    cost = ryc.cost(Y_test)
```

The predictions variable `Y_pred` has dimensions given by ($n_{features}\times \Omega_{test}$), where $n_{features}$ is the number indicated in the last layer nad $\Omega_{test}$ is the number of colums of `X_test`. Then, it can be convenient to transpose the data `Y_pred = Y_pred.T` for later usage such that the `Y_pred` is organanised in columns.


## Examples
A tutorial for different typical problem are included in the `tutorial` Jupiter notebook.
