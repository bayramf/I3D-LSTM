import os
import sys
sys.path.append('../../')
import time
import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_test
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
classes = ["car", "cyclist", "none", "pedestrian"]
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 20, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 4, 'The num of class')
FLAGS = flags.FLAGS

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    model_save_dir = './datasets1/seq=20_step=1/Datensatz_HHN_2_50_V4_obs'
    model_save_weights = './datasets1/seq=20_step=1/Datensatz_HHN_2_50_V4_obs/weights/rgb'

    with tf.Graph().as_default():
        rgb_images_placeholder, _, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels
                        )

        with tf.variable_scope('RGB'):
            logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits_LSTM',
                                name='inception_i3d'
                                )(rgb_images_placeholder, is_training)
        norm_score = tf.nn.softmax(logit)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)

    ckpt = tf.train.get_checkpoint_state(model_save_weights)
    if ckpt and ckpt.model_checkpoint_path:
        print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("load complete!")
        
    #%% 
    filename = model_save_dir + '/train.list'
    lines=[]
    with open(filename) as myfile:
        for line in myfile:
            lines.append(line)
    max_steps = len(lines)
    predicts = []
    true = []
    for step in xrange(max_steps):
        if step > 0:
            step_inc = step*FLAGS.batch_size
        else:
            step_inc = step
        start_time = time.time()
        rgb_train_images, _, train_labels = input_data.read_clip_and_label(
                      batch_size=FLAGS.batch_size * gpu_num,
                      lines=lines[step_inc:step_inc+FLAGS.batch_size],
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      add_flow=False,
                      add_rgb=True
                      )
        predict, true_labs = sess.run([norm_score, labels_placeholder],
                            feed_dict={
                                        rgb_images_placeholder: rgb_train_images,
                                        labels_placeholder: train_labels,
                                        is_training: False
                                        })
        predicts.append(predict)
        lb.fit([0, 1, 2, 3])
        true.append(lb.transform(true_labs))
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        
    y_pred = np.vstack(predicts)
    y_true = np.vstack(true)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_df = pd.DataFrame(cm,
                          index = classes, 
                          columns = classes)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='g')
    plt.title('Prediction training (Grayscale images) \nAccuracy: {0:.3f}'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_save_dir + '/confusion_rgb_train.png')
    #%%
    filename = model_save_dir + '/test1.list'
    lines=[]
    with open(filename) as myfile:
        for line in myfile:
            lines.append(line)
    max_steps = len(lines)
    predicts = []
    true = []
    for step in xrange(max_steps):
        if step > 0:
            step_inc = step*FLAGS.batch_size
        else:
            step_inc = step
        start_time = time.time()
        rgb_train_images, _, train_labels = input_data.read_clip_and_label(
                      batch_size=FLAGS.batch_size * gpu_num,
                      lines=lines[step_inc:step_inc+FLAGS.batch_size],
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      add_flow=False,
                      add_rgb=True
                      )
        predict, true_labs = sess.run([norm_score, labels_placeholder],
                            feed_dict={
                                        rgb_images_placeholder: rgb_train_images,
                                        labels_placeholder: train_labels,
                                        is_training: False
                                        })
        predicts.append(predict)
        lb.fit([0, 1, 2, 3])
        true.append(lb.transform(true_labs))
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        
    y_pred = np.vstack(predicts)
    y_true = np.vstack(true)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_df = pd.DataFrame(cm,
                          index = classes, 
                          columns = classes)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='g')
    plt.title('Prediction test 1 (Grayscale images) \nAccuracy: {0:.3f}'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_save_dir + '/confusion_rgb_test1.png')
    #%%
    filename = model_save_dir + '/test2.list'
    lines=[]
    with open(filename) as myfile:
        for line in myfile:
            lines.append(line)
    max_steps = len(lines)
    predicts = []
    true = []
    for step in xrange(max_steps):
        if step > 0:
            step_inc = step*FLAGS.batch_size
        else:
            step_inc = step
        start_time = time.time()
        rgb_train_images, _, train_labels = input_data.read_clip_and_label(
                      batch_size=FLAGS.batch_size * gpu_num,
                      lines=lines[step_inc:step_inc+FLAGS.batch_size],
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      add_flow=False,
                      add_rgb=True
                      )
        predict, true_labs = sess.run([norm_score, labels_placeholder],
                            feed_dict={
                                        rgb_images_placeholder: rgb_train_images,
                                        labels_placeholder: train_labels,
                                        is_training: False
                                        })
        predicts.append(predict)
        lb.fit([0, 1, 2, 3])
        true.append(lb.transform(true_labs))
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        
    y_pred = np.vstack(predicts)
    y_true = np.vstack(true)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_df = pd.DataFrame(cm,
                         index = classes, 
                         columns = classes)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='g')
    plt.title('Prediction test 2 (Grayscale images) \nAccuracy: {0:.3f}'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_save_dir + '/confusion_rgb_test2.png')
    #%%
    filename = model_save_dir + '/test3.list'
    lines=[]
    with open(filename) as myfile:
        for line in myfile:
            lines.append(line)
    max_steps = len(lines)
    predicts = []
    true = []
    for step in xrange(max_steps):
        if step > 0:
            step_inc = step*FLAGS.batch_size
        else:
            step_inc = step
        start_time = time.time()
        rgb_train_images, _, train_labels = input_data.read_clip_and_label(
                      batch_size=FLAGS.batch_size * gpu_num,
                      lines=lines[step_inc:step_inc+FLAGS.batch_size],
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      add_flow=False,
                      add_rgb=True
                      )
        predict, true_labs = sess.run([norm_score, labels_placeholder],
                            feed_dict={
                                        rgb_images_placeholder: rgb_train_images,
                                        labels_placeholder: train_labels,
                                        is_training: False
                                        })
        predicts.append(predict)
        lb.fit([0, 1, 2, 3])
        true.append(lb.transform(true_labs))
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        
    y_pred = np.vstack(predicts)
    y_true = np.vstack(true)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_df = pd.DataFrame(cm,
                         index = classes, 
                         columns = classes)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='g')
    plt.title('Prediction test 3 (Grayscale images) \nAccuracy: {0:.3f}'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_save_dir + '/confusion_rgb_test3.png')
    
    #%%
    # filename = model_save_dir + '/test4.list'
    # lines=[]
    # with open(filename) as myfile:
    #     for line in myfile:
    #         lines.append(line)
    # max_steps = len(lines)
    # predicts = []
    # true = []
    # for step in xrange(max_steps):
    #     if step > 0:
    #         step_inc = step*FLAGS.batch_size
    #     else:
    #         step_inc = step
    #     start_time = time.time()
    #     rgb_train_images, flow_train_images, val_labels, _, _, _ = input_data.read_clip_and_label(
    #                   batch_size=FLAGS.batch_size * gpu_num,
    #                   lines=lines[step_inc:step_inc+FLAGS.batch_size],
    #                   num_frames_per_clip=FLAGS.num_frame_per_clib,
    #                   crop_size=FLAGS.crop_size,
    #                   shuffle=True,
    #                   add_flow=False,
    #                   step=step
    #                   )
    #     predict, true_labs = sess.run([norm_score, labels_placeholder],
    #                        feed_dict={
    #                                     rgb_images_placeholder: rgb_train_images,
    #                                     labels_placeholder: val_labels,
    #                                     is_training: False
    #                                     })
    #     predicts.append(predict)
    #     lb.fit([0, 1, 2, 3])
    #     true.append(lb.transform(true_labs))
    #     duration = time.time() - start_time
    #     print('Step %d: %.3f sec' % (step, duration))

    # y_pred = np.vstack(predicts)
    # y_true = np.vstack(true)
    # cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    # cm_df = pd.DataFrame(cm,
    #                      index = classes, 
    #                      columns = classes)
    # plt.figure(figsize=(7,6))
    # sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='g')
    # plt.title('Prediction test 4 (Grayscale images) \nAccuracy: {0:.3f}'.format(accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))))
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.savefig(model_save_dir + '/confusion_rgb_test4.png')
    
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
