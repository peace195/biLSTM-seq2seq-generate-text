#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import time
from collections import OrderedDict

import tensorflow as tf

from utils import *
from model import Seq2SeqModel
import codecs
import numpy as np

trainX = []
trainY = []
trainX_len = []
trainY_len = []
word_dict = {}
embedding = []
SEQ_MAX_LEN = 100
f_vec = codecs.open('./data/glove.6B.50d.txt', 'r', 'utf-8')
idx = 0
for line in f_vec:
    if len(line) < 50:
        continue
    else:
        component = line.strip().split(' ')
        word_dict[component[0].lower()] = idx
        word_vec = list()
        for i in range(1, len(component)):
            word_vec.append(float(component[i]))
        embedding.append(word_vec)
        idx = idx + 1
f_vec.close()
unk_id = word_dict['unk']
src_vocab_size = len(word_dict)
start_id = src_vocab_size
end_id = src_vocab_size + 1
word_dict['start_id'] = start_id
embedding.append([0.] * len(embedding[0]))
word_dict['end_id'] = end_id
embedding.append([0.] * len(embedding[0]))
word_dict_rev = {v: k for k, v in word_dict.items()}
emb_dim = 50
src_vocab_size = src_vocab_size + 2
fout = open("./data/lp_train.txt", "r")

for line in fout:
    Y = []
    words = line.strip().replace("\t", " ").split()
    for word in words[:-2]:
        try:
            Y.append(word_dict[word])
        except KeyError:
            continue
    try:
        trainX.append([word_dict[words[-2]], word_dict[words[-1]]])
        trainX_len.append(2)
    except KeyError:
        continue
    trainY.append(Y)
    trainY_len.append(len(Y))

fout.close()
src_len = len(trainX)

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 50, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 4, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 30000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 30000, 'Target vocabulary size')
# NOTE(sdsuo): We used the same vocab for source and target
tf.app.flags.DEFINE_integer('vocab_size', src_vocab_size, 'General vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 10000, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('max_seq_length', 100, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1150000, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'model', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('summary_dir', 'model/summary', 'Path to save model summary')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_boolean('bidirectional', True, 'Use bidirectional encoder')
tf.app.flags.DEFINE_string('train_mode', 'ground_truth', 'Decode helper to use for training')
tf.app.flags.DEFINE_float('sampling_probability', 0.1, 'Probability of sampling from decoder output instead of using ground truth')

# TODO(sdsuo): Make start token and end token more robust
tf.app.flags.DEFINE_integer('start_token', start_id, 'Start token')
tf.app.flags.DEFINE_integer('end_token', end_id, 'End token')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS


def load_or_create_model(sess, model, saver, FLAGS):
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters...')
        model.restore(sess, saver, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Created new model parameters...')
        sess.run(tf.global_variables_initializer())


def train(embedding):
    config_proto = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config_proto) as sess:
        # Build the model
        config = OrderedDict(sorted(FLAGS.__flags.items()))
        model = Seq2SeqModel(config, 'train')

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a saver
        # Using var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list=None)

        # Initiaize global variables or reload existing checkpoint
        load_or_create_model(sess, model, saver, FLAGS)

        # Load word2vec embedding
        embedding = np.array(embedding)
        model.init_vars(sess, embedding=embedding)

        step_time, loss = 0.0, 0.0

        print('Training...')
        for epoch_idx in range(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print('Training is already complete.',
                      'Current epoch: {}, Max epoch: {}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break


            # Prepare batch training data
            # source: [batch_size, time_steps]: keywords + SEP + previous sentences
            # source_lens: [batch_size]: length of source
            # target: [batch_size, time_steps]: current sentence
            # target_lens: [batch_size]: length of target
            for i in range(int(src_len/FLAGS.batch_size)):
                if i * FLAGS.batch_size > src_len:
                    start = src_len - FLAGS.batch_size
                else:
                    start = i * FLAGS.batch_size
                step_loss, summary = model.train(
                    sess,
                    encoder_inputs=trainX[start : start + FLAGS.batch_size],
                    encoder_inputs_length=trainX_len[start : start + FLAGS.batch_size],
                    decoder_inputs=trainY[start : start + FLAGS.batch_size],
                    decoder_inputs_length=trainY_len[start : start + FLAGS.batch_size]
                )

                loss += float(step_loss) / FLAGS.display_freq

                # Display information
                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")


                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(),
                          'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time)

                    loss = 0

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print('Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, saver, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                              indent=2)

            # Increase the epoch index of the model
            model.increment_global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))


        print('Saving the last model')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, saver, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                  indent=2)

    print('Training terminated')


def main(_):
    train(embedding)


if __name__ == '__main__':
    tf.app.run()
