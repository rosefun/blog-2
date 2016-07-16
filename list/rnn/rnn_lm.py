import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from rnnnumpy import RNNNumpy
from rnntheano import RNNTheano

class RNNLM:
    def __init__(self):
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.index_to_word = None
        self.word_to_index = None
        self.model = None

    def tokenize_data(self, n = -1):
        # download dependent nltk resources if you havn't.
        # nltk.download('punkt')

        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading sentences from gutenberg corpus ..."
        from nltk.corpus import gutenberg
        tokenized_sentences = []
        for s in gutenberg.sents('austen-emma.txt'):
            tokenized_sentences.append([self.sentence_start_token] + s[1:-1] + [self.sentence_end_token])
        print "Parsed %d sentences." % (len(tokenized_sentences))

        if n > 0:
            tokenized_sentences = tokenized_sentences[:n]

        # count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())

        self.vocabulary_size = int(len(word_freq.items()) * 0.95)

        # get the most common words, treat others words as unknown.
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        print "Using vocabulary size %d." % self.vocabulary_size
        print "The least frequent word is '%s' and appeared %d times." % \
              (vocab[-1][0], vocab[-1][1])
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(self.unknown_token)
        self.word_to_index = dict([(w,i) for i,w in enumerate(self.index_to_word)])

        # replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index
                                      else self.unknown_token for w in sent]

        # create training data
        x_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]]
                             for sent in tokenized_sentences])
        y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]]
                             for sent in tokenized_sentences])

        print ""
        print "Example sentence: '%s'" % tokenized_sentences[0]
        print "By word indexes: '%s'" % \
              [self.word_to_index[w] for w in tokenized_sentences[0]]

        return (x_train, y_train)

    def train_numpy(self, x_train, y_train, iterations):
        self.model = RNNNumpy(word_dim = self.vocabulary_size,
                              hidden_dim = 100, bptt_truncate = 4)
        self.model.sgd(x_train, y_train, 0.01, iterations)

    def train_theano(self, x_train, y_train, iterations):
        self.model = RNNTheano(word_dim = self.vocabulary_size,
                               hidden_dim = 100, bptt_truncate = 4)
        self.model.sgd(x_train, y_train, 0.01, iterations)

    def generate_sentence(self):
        # repeat until we get an end token
        sentence_start_idx = self.word_to_index[self.sentence_start_token]
        sentence_end_idx = self.word_to_index[self.sentence_end_token]
        unknown_word_idx = self.word_to_index[self.unknown_token]
        # start the sentence with the start token
        new_sentence = [sentence_start_idx]
        while new_sentence[-1] != sentence_end_idx:
            next_word_probs = self.model.forward_propagation(new_sentence)
            sampled_word = unknown_word_idx
            # skip unknown words
            while sampled_word == unknown_word_idx or \
                  sampled_word == sentence_start_idx:
                samples = np.random.multinomial(1, next_word_probs[0])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        return new_sentence

    def generate_sentences(self, num_sentences, min_length):
        for i in xrange(num_sentences):
            sent = []
            # We want long sentences, not sentences with one or two words
            while len(sent) < min_length:
                sent = self.generate_sentence()
                sent_str = [self.index_to_word[x] for x in sent[1:-1]]
            print " ".join(sent_str).encode('utf-8')
            print ""

if __name__ == "__main__":

    rnnlm = RNNLM()
    x_train, y_train = rnnlm.tokenize_data(200)
    rnnlm.train_theano(x_train, y_train, iterations = 100)
    rnnlm.generate_sentences(10, 7)
