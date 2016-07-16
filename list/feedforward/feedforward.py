"""feedforward.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def delta(a, y):
        """Return the error delta dC/dz from the output layer.  Note that the
        derivative is proportional to error.
        see http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights()
        self.cost=CrossEntropyCost

    def init_weights(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def forward_propagation(self, x):
        """Return the output of the network given ``x`` as input."""
        a = x
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate, l2, evaluation_data=None):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``l2``. It returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, learning_rate, l2, len(training_data))
            print "Epoch %s training complete" % j

            # cost on training data
            cost = self.total_cost(training_data, l2)
            training_cost.append(cost)
            print "Cost on training data: {}".format(cost)

            # accuracy on training data
            accuracy = self.accuracy(training_data)
            training_accuracy.append(accuracy)
            print "Accuracy on training data: {} / {}".format(accuracy, \
                    len(training_data))

            if evaluation_data:
                # cost on evaluation data
                cost = self.total_cost(evaluation_data, l2)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)

                # accuracy on evaluation data
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), len(evaluation_data))
            print
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, l2, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``learning_rate`` is the
        learning rate, ``l2`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        dcdb = [np.zeros(b.shape) for b in self.biases]
        dcdw = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_dcdb, delta_dcdw = self.bp(x, y)
            dcdb = [nb+dnb for nb, dnb in zip(dcdb, delta_dcdb)]
            dcdw = [nw+dnw for nw, dnw in zip(dcdw, delta_dcdw)]
        self.weights = [(1-learning_rate*(l2/n))*w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, dcdw)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, dcdb)]

    def bp(self, x, y):
        """Return a tuple ``(dcdb, dcdw)`` representing the
        gradient for the cost function C_x in each layer"""
        dcdb = [np.zeros(b.shape) for b in self.biases]
        dcdw = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if(i == len(self.biases) - 1):
                activation = softmax(z)
            else:
                activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        dcdz = (self.cost).delta(activations[-1], y)
        dcdb[-1] = dcdz
        dcdw[-1] = np.dot(dcdz, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            dcdz = np.dot(self.weights[-l+1].transpose(), dcdz) * sp
            dcdb[-l] = dcdz
            dcdw[-l] = np.dot(dcdz, activations[-l-1].transpose())
        return (dcdb, dcdw)

    def accuracy(self, data):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        """

        results = [(np.argmax(self.forward_propagation(x)), np.argmax(y))
                   for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, l2):
        """Return the total cost for the data set ``data``. """
        cost = 0.0
        for x, y in data:
            a = self.forward_propagation(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(l2/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    """The softmax function"""
    sf = np.exp(z)
    sf = sf/np.sum(sf, axis=0)
    return sf

if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = Network([784, 30, 10])
    net.sgd(training_data, 30, 10, 0.5,
            l2 = 5.0,
            evaluation_data=validation_data)
