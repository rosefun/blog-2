import numpy as np
import theano as theano
import theano.tensor as T
import operator
from datetime import datetime
import os

class LSTMTheano:

    def __init__(self, word_dim, hidden_dim, bptt_truncate):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t_prev, U, V, W):
            z_t = T.nnet.hard_sigmoid(U[0][:,x_t] + W[0].dot(s_t_prev))
            r_t = T.nnet.hard_sigmoid(U[1][:,x_t] + W[1].dot(s_t_prev))
            c_t = T.tanh(U[2][:,x_t] + W[2].dot(s_t_prev * r_t))
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev

            o_t = T.nnet.softmax(V.dot(s_t))

            return [o_t[0], s_t]

        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[self.U, self.V, self.W],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Gradients
        dU = T.grad(o_error, self.U)
        dV = T.grad(o_error, self.V)
        dW = T.grad(o_error, self.W)

        # Assign functions
        self.forward_propagation = theano.function([x], [o[-1], s[-1]])
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [],
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])

    def total_cost(self, xs, ys):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in ys])
        L = np.sum([self.ce_error(x,y) for x,y in zip(xs, ys)])
        return L/float(num_words)

    def sgd(self, x_train, y_train, learning_rate, epochs):
        losses = []
        for j in range(epochs):
            # for each training example...
            for i, (x, y) in enumerate(zip(x_train, y_train)):
                # update parameters according to gradients and learning rate
                self.sgd_step(x, y, learning_rate)

            # saving model parameters
            time = datetime.now().strftime('%Y%m%d%H%M%S')
            model_file = "./output/rnn-theano.model.%d.npz" % (j)
            if not os.path.exists(os.path.dirname(model_file)):
                os.makedirs(os.path.dirname(model_file))
            U, V, W = self.U.get_value(), self.V.get_value(), self.W.get_value()
            np.savez(model_file, U = U, V = V, W = W)
            print "Saved model parameters to %s." % model_file

            # evaluate the loss
            loss = self.total_cost(x_train, y_train)
            losses.append(loss)
            print "%s: Epoch %s training complete. Cost on training data: %f" % \
                  (time, j, loss)
            # adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1] > losses[-2] and learning_rate > 0.0001):
                learning_rate = learning_rate * 0.5
                print "Set learning rate to %f" % learning_rate


def softmax(z):
    """The softmax function"""
    sf = np.exp(z)
    sf = sf/np.sum(sf, axis=0)
    return sf
