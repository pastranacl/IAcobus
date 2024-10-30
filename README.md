# Neural Network from Scratch

This is an exercise to build a full neural network from scratch. It use Numpy vectorisation for an optimal performance.
The main code is contained in the neurnet class.

## Instructions
The implementation approach is inspired by the paradigm of TensorFlow + Keras.


1. **Preprocessing of arrays**

2. **Create object** The neural network is created by specifiying the number of inputs, the topology,  the activation functions for each layer, and hte cost function. The topology of the network as well as the associated activation functions for each layer are provided as arrays. The following example creates the network `ryc`, composed of three hidden layers and one input variable´. The two first layers have 64 $\times$ ReLU activation functions, and the last one is a linear function. The cost function is the mean-squared error.

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

3. **Training**

```
cost = ryc.train(X, Y,
                 num_epochs=5000,
                 batch_size=10,
                 learning_rate=1e-6,
                 verbose=True)
```

The `verbose` variable is boolean as prints the epoch number and the cost during training.


4. **Forward propagation**

## Examples
Implementation examples are included in the Jupiter notebook.
