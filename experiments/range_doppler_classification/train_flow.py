import os
import sys
sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from tensorflow.python import pywrap_tensorflow
import random

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 20, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 4, 'The num of class')
FLAGS = flags.FLAGS

model_save_dir = './datasets1/seq=20_step=1/Datensatz_HHN_2_50_V4_obs'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    flow_pre_model_save_dir = "/home/project/I3D/I3D/checkpoints/flow_scratch"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels,
                        FLAGS.flow_channels
                        )

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
        opt_flow = tf.train.AdamOptimizer(learning_rate)
        #opt_stable = tf.train.MomentumOptimizer(learning_rate, 0.9)

        with tf.variable_scope('Flow'):
            flow_logit, _ = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits_LSTM'
                                    )(flow_images_placeholder, is_training)
        flow_loss = tower_loss(
                                flow_logit,
                                labels_placeholder
                                )
        accuracy = tower_acc(flow_logit, labels_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            flow_grads = opt_flow.compute_gradients(flow_loss)
            apply_gradient_flow = opt_flow.apply_gradients(flow_grads, global_step=global_step)
            train_op = tf.group(apply_gradient_flow)
            null_op = tf.no_op()

        # Create a saver for loading trained checkpoints.
        flow_variable_map = {}

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow'and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits_LSTM':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        # Create summary writter
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('flow_loss', flow_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()
    # load pre_train models
    ckpt = tf.train.get_checkpoint_state(flow_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        flow_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    train_writer = tf.summary.FileWriter('./visual_logs/train_flow_scratch_20000_6_64_0.0001_decay', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test_flow_scratch_20000_6_64_0.0001_decay', sess.graph)
    #training_losses = []
    #training_accs = []
    #%%
    filename = model_save_dir + '/train.list'
    lines=[]
    with open(filename) as myfile:
        for line in myfile:
            lines.append(line)
            
    for epoch in xrange(FLAGS.epochs):
        print("Epoch " + str(epoch))
        print("Training")
        #training_acc = 0
        #training_loss = 0
                
        print("Shuffling data")
        lines = random.sample(lines, len(lines))
        max_steps = len(lines)//FLAGS.batch_size

        # generator = input_data.read_clip_and_label(
        #               batch_size=FLAGS.batch_size * gpu_num,
        #               files=lines,
        #               num_frames_per_clip=FLAGS.num_frame_per_clib,
        #               crop_size=FLAGS.crop_size,
        #               shuffle=True,
        #               add_flow=False
        #               )
        
        for step in xrange(max_steps):
            if step > 0:
                step_inc = step*FLAGS.batch_size
            else:
                step_inc = step
            start_time = time.time()
            _, flow_train_images, train_labels = input_data.read_clip_and_label(
                          batch_size=FLAGS.batch_size * gpu_num,
                          lines=lines[step_inc:step_inc+FLAGS.batch_size],
                          num_frames_per_clip=FLAGS.num_frame_per_clib,
                          crop_size=FLAGS.crop_size,
                          add_flow=True,
                          add_rgb=False
                          )
            # rgb_train_images, _, train_labels, _, _ = next(generator)
            sess.run(train_op, feed_dict={
                          flow_images_placeholder: flow_train_images,
                          labels_placeholder: train_labels,
                          is_training: True
                          })
            
            duration = time.time() - start_time
            print('Step %d of %s: %.3f sec' % (step, max_steps, duration))
            #training_loss += loss_rgb
            #training_acc += acc
            if step % 100 == 0 and step > 0:
                print("Epoch " + str(epoch))
                summary, acc, loss_flow= sess.run(
                                [merged, accuracy, flow_loss],
                                feed_dict={flow_images_placeholder: flow_train_images,
                                            labels_placeholder: train_labels,
                                            is_training: False
                                          })
                print("accuracy: " + "{:.5f}".format(acc))
                print("flow_loss: " + "{:.5f}".format(loss_flow))
                # print("Average loss at step", step,
                #       "for last 100 steps:", training_loss/100)
                # print("Average acc at step", step,
                #       "for last 100 steps:", training_acc/100)
                # training_losses.append(training_loss/100)
                # training_accs.append(training_acc/100)
                # training_loss = 0
                # training_acc = 0
                
            if (step + 1) == max_steps:
                saver.save(sess, os.path.join(model_save_dir + '/weights/flow', 'final_weights'), global_step=step)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
