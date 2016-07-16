import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime

class RNNNumpy:

    def __init__(self, word_dim, hidden_dim, bptt_truncate):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim),
                                   np.sqrt(1./word_dim),
                                   (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),
                                   np.sqrt(1./hidden_dim),
                                   (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),
                                   np.sqrt(1./hidden_dim),
                                   (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # save all hidden states and outpts at each time step
        # because need them during back propagation.
        # add one additional 0s as the initial hidden state
        s = np.zeros(self.hidden_dim)
        o = np.zeros(self.word_dim)
        for t in range(len(x)):
            # Note that we are indxing U by x[t]. This is the same as
            # multiplying U with a one-hot vector.
            s = np.tanh(self.U[:,x[t]] + self.W.dot(s))
            o = softmax(self.V.dot(s))
        return [o, s]

    def bptt(self, x, y):
        # forward pass
        # save all hidden states and outpts at each time step
        # because need them during back propagation.
        # add one additional 0s as the initial hidden state
        s = np.zeros((len(x) + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((len(x), self.word_dim))
        for t in range(len(x)):
            # Note that we are indxing U by x[t]. This is the same as
            # multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        # backward pass
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        # dLdy = o - y
        dLdy = o
        dLdy[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(len(y))[::-1]:
            dLdV += np.outer(dLdy[t], s[t].T)
            # Initial delta calculation
            dLdz = self.V.T.dot(dLdy[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(dLdz, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += dLdz
                # Update delta for next step
                dLdz = self.W.T.dot(dLdz) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def sgd(self, x_train, y_train, learning_rate, epochs):
        losses = []
        for j in range(epochs):
            # for each training example...
            for i, (x, y) in enumerate(zip(x_train, y_train)):
                dLdU, dLdV, dLdW = self.bptt(x, y)
                # update parameters according to gradients and learning rate
                self.U -= learning_rate * dLdU
                self.V -= learning_rate * dLdV
                self.W -= learning_rate * dLdW

            # evaluate the loss
            loss = self.total_cost(x_train, y_train)
            losses.append(loss)
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Epoch %s training complete. Cost on training data: %f" %\
                  (time, j, loss)
            # adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1] > losses[-2]):
                learning_rate = learning_rate * 0.5
                print "Set learning rate to %f" % learning_rate

    def total_cost(self, xs, ys):
        L = 0
        # For each sentence...
        for i in xrange(len(ys)):
            o, s = self.forward_propagation(xs[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[ys[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))

        # Divide the total loss by the number of training examples
        N = np.sum((len(y) for y in ys))
        return L/N

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

def softmax(z):
    """The softmax function"""
    sf = np.exp(z)
    sf = sf/np.sum(sf, axis=0)
    return sf
