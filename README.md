<h1 align="center">
    <img src="imgs/.iacobus_logo.svg" alt="IAcobus"  width="350">
</h1><br>

**IAcobus** is a fully functional library for building fully connected neural networks. It is named after the renowned Spanish neuroscientist Santiago Ramón y Cajal, who was awarded the Nobel Prize in Physiology or Medicine in 1906, and the first person to identify the neurons of the brain. The name “Santiago” itself is derived etymologically from *Saint Iacobus* (Saint James in English).

This project began as an exercise to master the implementation of neural networks from scratch, inspired by Richard Feynman's famous overly used quote: *"What I cannot create, I do not understand."* The library includes batch backpropagation training, and is optimised using Numpy vectorisation for improved speed. However, the design prioritises clarity of the underlying steps rather than performance. For a full mathematical explanation of the principles behind neural networks, please refer to the publication *"Mathematical Basis of Neural Networks for Curious Physicists"* by C.L. Pastrana (2024).

## Installation
The code uses Python 3.12, Numpy 1.26.4. As accesory libraries: dill ($\sim$ pickle on steroids) and tqdm to monitor the training progress.
Installation files are provided in the `setup` folder to install via pip.
Matplotlib is used in the tutorial for the reppresentation of tdata; the MNIST database is taken from the Scikit-learn library. These two libraries are not needed in the core code.

## Instructions
The implementation approach is inspired by the paradigm of TensorFlow + Keras.

1. **Create neural network object:** The neural network is created by specifiying the number of inputs, the topology,  the activation functions for each layer, and hte cost function. The topology of the network as well as the associated activation functions for each layer are provided as arrays. The following code creates the network `ryc`, composed of three hidden layers and one input variable. The two first layers have 64 $\times$ ReLU activation functions, and the last one is a linear function. The cost function is the mean-squared error.

```
    from iacobus import IAcobus

    n_inputs = 1
    costf = 'mse'
    topology = [64, 64, 1]
    activation_funcs = ['relu',
                        'relu',
                        'linear']

    ryc = IAcobus(n_inputs,
                    topology,
                    activation_funcs,
                    cost_func = costf)
```

| Argument        | Description                |
|   ---            | ---                        |
| `n_inputs`          | int. Number of features |
| `topology` | Array containing the number of neurons on each layer |
| `activation_funcs` | Array of strings for the activation function of each layer. The dimensions must coincide with that of `topology`. The activation functions available are the Heavyside `heavyside`, the rectifying linear unit ReLU `relu`, the sigmoid `sigmoid`, the hyperbolic tagent `tanh`, and the softmax function `softmax`   |
| `cost_func`| String indicating the cost function to use. The options are the mean-squared error `mse`, binary cross entropy `bin-cross-ent` and  multiple cross entropy `cross-entropy|


In the current implementation, the `softmax` layer can only be used in the last layer and ìn combination with the  multiple cross entropy cost.

2. **Preprocessing of data:** IAcobus uses column vectors by default. That is, each row is a feature, with different observations organanised in the columns. This organisation is different to the one typically used in `.csv` or tab formated files, such that the data would need to be transposed before being feed to IAcobus.

For classification problems, the labels need to be converted to one-hot-encoded vectors. If `Y` is an array of labels for each observation, we can create a one-hot reppresentation with:

```
     Y_one_hot = ryc.one_hot_encoding(Y, n_classes)
```
where `n_classes` is the total number of classes on our problem.

To split the data in training and test sets, IAcobus includes the function:

```
   X_train, Y_train, X_test, Y_test = ryc.split_data_train_test(X, Y, p_train=0.7)
```
where `p_train` defines the fraction of the data samples to use for training. Note that a second call to this function with the non-training data can be used to split the data in cross-validation and test sets.


3. **Training:** Initially, the weights $W$ are initialised with random numbers (using He-initilisation). Biases $\vec{b}$ are initilised to zero. Labeled data can be used to train the network, such that the optimal $W$ and $b$ for each neuron of the network is determined as those minimising the cost function. The following code uses the matrix $X$ as input and $Y$ labeled data.


```
    cost = ryc.train(X_train, Y_train,
                    num_epochs=5000,
                    batch_size=10,
                    learning_rate=1e-6,
                    algorithm='adam',
                    verbose=True,
                    **kwargs)
```

The `cost` is an array with the cost for each epoch, and this is useful for reppresenting training curves.

| Argument        | Description                |
|   ---            | ---                        |
| `num_epochs`     | int. The number of optimisation steps, considering as optimisation step the full pass to the data set |
| `batch_size`      | int. Samples to use. For a total number of $\Omega$ samples, and batch size $\omega$, the total number of batches is simply $\Omega/\omega$. Since this library works on the CPU, fitting the data on memory is not a problem, and then this hyperparameter is not so critical. |
| `learning_rate` | float. Initial learning rate for the optimiser. |
| `algorithm` | string. Optimisation algorithm to use. The options are Adam `adam` (default), the recent ADOPT algorithm `adopt`, gradient descendent with momentum `gdm`, and basic gradient descendent `gd`.  |
| `verbose` | bool. Prints the epoch number and the cost during training. |
| `**kwargs` | floats. Additional parameters to pass to the minimisation algorithm. In particular, the $\beta_1$ and $\beta_2$ of the Adam algorithm can be selected as `beta1=0.9` and `beta2=0.999`. The parameter $\beta$ in the gradient descendent with momentum is set as $\beta=0.9$. Choosing the Adam algorithm and indicating `beta1=0` leads to the RMSProp method.  |

4. **Forward propagation:** To obtain predictions form an input array `X_test` the `forward` function is called:

```
    Y_pred = ryc.forward(X_test)
```

If test or cross-validation data are available, the associated cost can be calculated using the `cost function:`
```
    cost = ryc.cost(Y_test)
```

The predictions variable `Y_pred` has dimensions given by ($n_{features}\times \Omega_{test}$), where $n_{features}$ is the number of neurons of the last layer, and $\Omega_{test}$ is the number of observations, and hence colums in `X_test`. Then, it can be convenient to transpose the data `Y_pred = Y_pred.T` for later usage such that the `Y_pred` is organanised in columns.

5. **Saving the network**. It is convenient to save the network object to continue training the data later or to use an alredy trained network:

```
    ryc.export("trained_network.pkl")
```

The saved network can be loaded with the `export` function in a completely new and independent instance as:
```
    from iacobus import IAcobus

    rycTrained = IAcobus.load("trained_network.pkl")
```

This allow us to continue training the network or to call it for feedforward to obtain predictions. However, the topology of the network is created at inialisation and can not be directly modified.

## Examples
A full tutorial showing the operation of the library to tackle different problems, designed with a minimalist and mathematical perspective, are included in the Jupiter notebook `tutorial.ipynb`.

