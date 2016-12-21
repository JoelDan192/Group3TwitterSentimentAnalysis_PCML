# License from the code provided by the Tensorflow tutorial for Word2Vec at:
# http://www.tensorflow.org/tutorials/word2vec/
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ** We extended the tutorial code as a class to be used in our existing system
# with CNNs in tensorflow **
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class WordToVec():
    def __init__(self, corpus=None, vocabulary_size = 60000,
                batch_size = 128,
                embedding_size = 128,  # Dimension of the embedding vector.
                skip_window = 1,       # How many words to consider left and right.
                num_skips = 2,         # How many times to reuse an input to generate a label.
                num_steps = 300001,
                valid_size = 16,     # Random set of words to evaluate similarity on.
                valid_window = 100,  # Only pick dev samples in the head of the distribution.
                num_sampled = 64):    # Number of negative examples to sample.
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.data_index = 0
        self.num_steps = num_steps
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.num_sampled = num_sampled
        self.data = list()
        self.count = None
        self.dictionary = dict()
        self.reverse_dictionary = dict()
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.final_embeddings = None
        self.__graph = None
        self.__loss = None
        self.__optimizer = None
        self.__similarity = None
        self.__normalized_embeddings = None
        self.__valid_embeddings = None
        self.__init = None
        self.build_dataset(corpus)
        print('Starting Word2Vec generation...')
        print('Most common words (+UNK)', self.count[:5])
        print('Sample data', self.data[:10], [self.reverse_dictionary[i] for i in self.data[:10]])
        self.__build_graph()


    def build_dataset(self, words):
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        print('Nbr of words passed: ',len(self.count))
        #dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        #data = list()
        unk_count = 0
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self,batch_size, num_skips, skip_window):
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        assert span > 1
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)

        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


    def embedding_for(self, word):
        if word in self.dictionary:
            return self.final_embeddings[self.dictionary[word]]
        else:
            return np.zeros(self.final_embeddings.shape[1], dtype=np.float32)


    def __build_graph(self):
        self.__graph = tf.Graph()

        with self.__graph.as_default():
            # Input data.
            self.__train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.__train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.__train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            self.__loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, self.__train_labels,
                                self.num_sampled, self.vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            self.__optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.__loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.__normalized_embeddings = embeddings / norm
            self.__valid_embeddings = tf.nn.embedding_lookup(
                self.__normalized_embeddings, valid_dataset)
            self.__similarity = tf.matmul(
                self.__valid_embeddings, self.__normalized_embeddings, transpose_b=True)
            self.__init = tf.global_variables_initializer()




    def train(self):

        # Add variable initializer.
        #init = tf.global_variables_initializer()

        with tf.Session(graph=self.__graph) as session:
            # We must initialize all variables before we use them.
            self.__init.run(session=session)
            print("Initialized")

            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels = self.generate_batch(
                    self.batch_size, self.num_skips, self.skip_window)
                feed_dict = {self.__train_inputs: batch_inputs, self.__train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.__optimizer, self.__loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.__similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.reverse_dictionary[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            self.final_embeddings = self.__normalized_embeddings.eval()



    def plot_word_cloud(self, filename='tsne.png'):
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
            labels = [self.reverse_dictionary[i] for i in xrange(plot_only)]

            assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
            plt.figure(figsize=(18, 18))  # in inches
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

            plt.savefig(filename)

        except ImportError:
            print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
