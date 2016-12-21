#! /usr/bin/env python
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
import numpy as np
import os
import time
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import nltk
import sys
sys.path.append(os.getcwd()+"\\tensorflow-glove")
sys.path.append(os.getcwd())
import tf_glove
import word2vecSimple as word2vec
from helpers import *

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/twitter-datasets/train_pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/twitter-datasets/train_neg.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.05, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("hparams", "0.005,0.2 0.05,0.4 0.1,0.5 0.2,0.7", "dropout probability and l2 penalty combinations")
tf.flags.DEFINE_string("channels", "fromscratch", "Preferred channel options: glove, word2vec, fromscratch")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 110, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("iters_per_fold", 100, "number of steps per fold in crossvalidation")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file) # Build vocabulary
#x_text, y = x_text[:int(len(x_text)*0.65)], y[:int(len(y)*0.65)]
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

glove_matrix = np.array([])
word2vec_matrix = np.array([])

if 'word2vec' in FLAGS.channels:
    try:
        word2vec_matrix = np.load('word2vec_embeddings.npy')
    except FileNotFoundError:
        vocab_dict = vocab_processor.vocabulary_._mapping
        tweets = (nltk.wordpunct_tokenize(tweet) for tweet in x_text)
        corpus_w2vec = [w for tw in tweets for w in tw]
        w2vec_model = word2vec.WordToVec(corpus=corpus_w2vec, embedding_size=FLAGS.embedding_dim)
        w2vec_model.train()
        w2vec_model.plot_word_cloud(filename='tsne.png')
        word2vec_matrix = np.vstack((w2vec_model.embedding_for(word)
                                  for word in sorted(vocab_dict,key=lambda k: vocab_dict[k])))
        np.save('word2vec_embeddings.npy', word2vec_matrix)
if 'glove' in FLAGS.channels:
    try:
        glove_matrix = np.load('glove_embeddings.npy')
    except FileNotFoundError:
        vocab_dict = vocab_processor.vocabulary_._mapping
        #lists of tokens for GloVe embedding generator
        corpus_glove = (nltk.wordpunct_tokenize(tweet) for tweet in x_text)
        glove_model = tf_glove.GloVeModel(embedding_size=FLAGS.embedding_dim, context_size=10, min_occurrences=15,
                                        learning_rate=0.05, batch_size=512)
        glove_model.fit_to_corpus(corpus_glove)
        glove_model.train(num_epochs=50)

        #Generate embedding matrix from vocabulary
        glove_matrix = np.vstack((glove_model.embedding_for(word)
                                for word in sorted(vocab_dict,key=lambda k: vocab_dict[k])))
        np.save('glove_embeddings.npy', glove_matrix)
if 'fromscratch' in FLAGS.channels:
    print('Initializing word embeddings in the random unit cube')

hparams = [list(map(float, pair.split(','))) for pair in FLAGS.hparams.split()]
k_fold = len(hparams)
seed = 1
k_indices = build_k_indices(y.shape[0], k_fold, seed)

# Cross-validation and Training
# ==================================================

accs = []
for k, hparam in enumerate(hparams):
    lbda, dropout_prob = hparam
    test_idx = k_indices[k,:]
    row_idx = list(range(k_indices.shape[0]))
    train_idx = k_indices[row_idx[:k]+row_idx[k+1:],:].flatten()
    y_train, x_train = y[train_idx], x[train_idx, :]
    y_dev, x_dev = y[test_idx], x[test_idx, :]
    x_train, mean_x, std_x = standardize(x_train)
    x_dev, _, _ = standardize(x_dev, mean_x=mean_x, std_x=std_x)

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...

    acc = None
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                glove_embedding_matrix = glove_matrix,
                word2vec_embedding_matrix = word2vec_matrix,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                channels=list(map(lambda st: st.strip(), FLAGS.channels.split(","))),
                l2_reg_lambda=lbda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                #time_str = datetime.datetime.now().isoformat()
                print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                #train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                #time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                #if writer:
                #    writer.add_summary(summaries, step)
                return accuracy

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    accs.append(dev_step(x_dev, y_dev, writer=dev_summary_writer))
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % FLAGS.iters_per_fold == 0:
                    break

np.save('accs_cnn.npy', np.array(accs))
