#
# Based on the implementation from:
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
#
# We extended this code to work with arbitrary multichannel (original implementation
# only works with one fixed random channel) embeddings and added support
# for GloVe and Word2Vec. Other features we added include crossvalidation and evaluation on
# precomputed language model generated examples.
#
#================================================================================

import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, glove_embedding_matrix, word2vec_embedding_matrix, filter_sizes, num_filters, channels=['fromscratch'], l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.nbr_channels = len(channels)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        def get_matrix(channel):
            if channel=='word2vec':
                return tf.Variable(word2vec_embedding_matrix)
            elif channel=='glove':
                return tf.Variable(glove_embedding_matrix)
            elif channel=='fromscratch':
                return tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            else:
                sys.exit('Available channel options: word2vec, glove, fromscratch')


        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            matrices = (get_matrix(ch) for ch in channels)
            embedded_chars = (tf.nn.embedding_lookup(W, self.input_x) for W in matrices)
            self.embedded_chars_expanded = (tf.expand_dims(ebd, -1) for ebd in embedded_chars)
            self.embedded_chars_expanded = tf.concat(self.nbr_channels + 1, list(self.embedded_chars_expanded))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, self.nbr_channels, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
