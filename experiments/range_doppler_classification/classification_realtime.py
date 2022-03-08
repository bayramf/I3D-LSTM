import os
import sys
sys.path.append('../../')
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_test
import input_data
import math
from i3d import InceptionI3d
from utils import *
import numpy as np
from modules.mmwaveold.dataloader import DCA1000
import modules.mmwaveold.dsp as dsp
from modules.mmwaveold.dsp.utils import Window
import cv2
import datetime
# mmwave.dataloader.parse_raw_adc('adc_data_Raw_0.bin', 'adc_data.bin') to clean

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

# Radar configuration parameters
num_samples = 256
chirp_loops = 255
num_rx = 1

num_tx = 1
chirps = 1
middle = np.load('middle.npy')

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
    model_save_weights = './weights'

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
        
    #%% Radar configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    dca = DCA1000()
    label = []
    rd_maps = []
    label.append(0)
    np_arr_label = np.array(label).astype(np.int64)
    np_arr_label.reshape(1)
    while True:
        adc_data = dca.read()
        adc_data = adc_data.reshape((-4,1))[::4,:].ravel()
        frame = DCA1000.organize(adc_data, chirp_loops*chirps, num_rx, num_samples, model='1443')
        # First FFT with windowing
        range_data = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        # Second FFT with windowing
        fft2d, fft2d_no_log = dsp.doppler_processing(range_data, num_tx_antennas=num_tx, clutter_removal_enabled=False, window_type_2d=Window.HAMMING, accumulate=False)
        # Shifting the zero-frequency component to the center of the array
        range_doppler = np.fft.fftshift(fft2d_no_log, axes=2)
        # Getting log values
        #range_doppler = np.log2(np.abs(range_doppler))
        range_doppler = 20*np.log10(np.abs(range_doppler) / ((np.abs(range_doppler).max()).max()))
        # Sum over all Rx
        range_doppler = range_doppler.sum(1)
        # Rotate to get samples (distance) on x-axis and frequency (velocity) on y-axis
        range_doppler = np.rot90(range_doppler)
        range_doppler[126:129,:]= middle
        range_doppler[range_doppler < -58] = -58
        
		# Range doppler image processing
        range_doppler = (range_doppler - range_doppler.min()) / (range_doppler.max() - range_doppler.min())
        range_doppler = cv2.applyColorMap((range_doppler*255).astype(np.uint8), cv2.COLORMAP_JET)
        range_doppler = cv2.cvtColor(range_doppler, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(range_doppler, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 7)
        range_doppler = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        rd_maps.append(range_doppler)
        range_doppler_res = cv2.resize(range_doppler, (600, 600))
        
		# Classification
        if len(rd_maps)==20:
            rd_maps_testing = []
            test_images = input_data.data_process(
                          tmp_data=rd_maps,
                          crop_size=FLAGS.crop_size
                          )
            rd_maps_testing.append(test_images)
            rd_maps_rt = np.array(rd_maps_testing).astype(np.float32)
            predict, _ = sess.run([norm_score, labels_placeholder],
                                feed_dict={
                                            rgb_images_placeholder: rd_maps_rt,
                                            labels_placeholder: np_arr_label,
                                            is_training: False
                                            })
            
            if int(predict.argmax(axis=1))==0:
                print("car")
            elif int(predict.argmax(axis=1))==1:
                print("cyclist")
            elif int(predict.argmax(axis=1))==3:
                print("pedestrian")
            else:
                print("none")

            del rd_maps[0]
            
        fivem = cv2.putText(range_doppler_res,'5',(130,330), font, 1,(255,255,255),2,cv2.LINE_AA)
        tenm = cv2.putText(range_doppler_res,'10',(260,330), font, 1,(255,255,255),2,cv2.LINE_AA)
        fifteenm = cv2.putText(range_doppler_res,'15',(410,330), font, 1,(255,255,255),2,cv2.LINE_AA)
        twentym = cv2.putText(range_doppler_res,'20',(555,330), font, 1,(255,255,255),2,cv2.LINE_AA)
        x_unit = cv2.putText(range_doppler_res,'in m',(510,280), font, 1,(255,255,255),2,cv2.LINE_AA)
        onemetpersec = cv2.putText(range_doppler_res,'1',(10,205), font, 1,(255,255,255),2,cv2.LINE_AA)
        twometpersec = cv2.putText(range_doppler_res,'2',(10,110), font, 1,(255,255,255),2,cv2.LINE_AA)
        threemetpersec = cv2.putText(range_doppler_res,'3',(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
        monemetpersec = cv2.putText(range_doppler_res,'-1',(10,405), font, 1,(255,255,255),2,cv2.LINE_AA)
        mtwometpersec = cv2.putText(range_doppler_res,'-2',(10,510), font, 1,(255,255,255),2,cv2.LINE_AA)
        mthreemetpersec = cv2.putText(range_doppler_res,'-3',(10,595), font, 1,(255,255,255),2,cv2.LINE_AA)
        y_unit = cv2.putText(range_doppler_res,'in m/s',(80,595), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Range-Doppler', range_doppler_res)
        if cv2.waitKey(1) == ord('q'):# Press q to quit
            cv2.destroyAllWindows()
            dca.close()
            break

    
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
