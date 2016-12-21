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
#
# PLEASE SEE THE INSTRUCTIONS IN THE KAGGLE SECTION OF README.md
#
#================================================================================
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "./data/twitter-datasets/test_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("dev_set_file", "./data/twitter-datasets/dev_set.txt", "dev set generated with stanford language model")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_train:
    x_raw, ids = data_helpers.load_test_data(FLAGS.test_data_file, ftype='test')
    assert len(ids)==10000
    print("Received", str(len(x_raw)),"test examples")
    x_dev, y_dev = data_helpers.load_test_data(FLAGS.dev_set_file, ftype='dev')

else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
x_dev = np.array(list(vocab_processor.transform(x_dev)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches_test = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        batches_dev = data_helpers.batch_iter(list(x_dev), FLAGS.batch_size, 1, shuffle=False)


        # Collect the predictions here
        test_predictions = []
        dev_predictions = []
        for x_test_batch in batches_test:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            test_predictions = np.concatenate([test_predictions, batch_predictions])
        for x_dev_batch in batches_dev:
            batch_predictions = sess.run(predictions, {input_x: x_dev_batch, dropout_keep_prob: 1.0})
            dev_predictions = np.concatenate([dev_predictions, batch_predictions])

y_dev = np.array(list(map(int,y_dev)))

# Print accuracy if y_test is defined
if y_dev is not None:
    correct_predictions = float(sum(dev_predictions == y_dev))
    print("Total number of DEV examples: {}".format(len(y_dev)))
    print("DEV ACCURACY: {:g}".format(correct_predictions/float(len(y_dev))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((ids, [1 if pred==1 else -1 for pred in test_predictions]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Prediction"])
    writer.writerows(predictions_human_readable)
