from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import cv2
import random
import numpy as np
import time



def sample_data(ori_arr, num_frames_per_clip, sample_rate):
    ret_arr = []
    for i in range(int(num_frames_per_clip/sample_rate)):
        ret_arr.append(ori_arr[int(i*sample_rate)])
    return ret_arr


def get_data(filename, num_frames_per_clip, sample_rate, s_index=-1):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames)==0:
            print('DATA_ERRO: %s'%filename)
            return [], s_index
        if (len(filenames)-s_index) <= num_frames_per_clip:
            filenames = sorted(filenames)
            if len(filenames) < num_frames_per_clip:
                for i in range(num_frames_per_clip):
                    if i >= len(filenames):
                        i = len(filenames)-1
                    image_name = str(filename) + '/' + str(filenames[i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            else:
                for i in range(num_frames_per_clip):
                    image_name = str(filename) + '/' + str(filenames[len(filenames)-num_frames_per_clip+i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            return sample_data(ret_arr, num_frames_per_clip, sample_rate), s_index
    filenames = sorted(filenames)
    if s_index < 0:
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(int(num_frames_per_clip/sample_rate)):
        image_name = str(filename) + '/' + str(filenames[int(i*sample_rate)+s_index])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, s_index


def get_frames_data(filename, num_frames_per_clip, sample_rate, add_flow, add_rgb):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    if add_rgb:
        filename_i = os.path.join(filename, 'i')
        rgb_ret_arr, s_index = get_data(filename_i, num_frames_per_clip, sample_rate)
    s_index = 0
    if not add_flow:
        return rgb_ret_arr, [], s_index
    filename_x = os.path.join(filename, 'x')
    flow_x, _ = get_data(filename_x, num_frames_per_clip, sample_rate, s_index)
    flow_x = np.expand_dims(flow_x, axis=-1)
    filename_y = os.path.join(filename, 'y')
    flow_y, _ = get_data(filename_y, num_frames_per_clip, sample_rate, s_index)
    flow_y = np.expand_dims(flow_y, axis=-1)
    flow_ret_arr = np.concatenate((flow_x, flow_y), axis=-1)
    if add_rgb:
        return rgb_ret_arr, flow_ret_arr, s_index
    else:
        return [], flow_ret_arr, s_index


def get_frames(filename, s_index, num_frames_per_clip, crop_size, sample_rate, add_flow, position=-1):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    rgb_ret_arr, s_index = get_data(filename, num_frames_per_clip, sample_rate, int(s_index))
    rgb_ret_arr = data_process_pos(rgb_ret_arr, crop_size, int(position))
    if not add_flow:
        return rgb_ret_arr, [], s_index


def data_process_pos(tmp_data, crop_size, position):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
            if j == 0:
                if position==-1:
                    crop_x = random.randint(0, int(img.shape[0] - crop_size))
                    crop_y = random.randint(0, int(img.shape[1] - crop_size))
                elif position==0:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = 0
                elif position==1:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int((img.shape[1] - crop_size) / 2)
                else:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int(img.shape[1] - crop_size)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
            if j == 0:
                if position==-1:
                    crop_x = random.randint(0, int(img.shape[0] - crop_size))
                    crop_y = random.randint(0, int(img.shape[1] - crop_size))
                elif position==0:
                    crop_x = 0
                    crop_y = int((img.shape[1] - crop_size) / 2)
                elif position==1:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int((img.shape[1] - crop_size) / 2)
                else:
                    crop_x = int(img.shape[0] - crop_size)
                    crop_y = int((img.shape[1] - crop_size) / 2)

        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)
    return img_datas

def data_process(tmp_data, crop_size):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            # crop_x = random.randint(0, int(img.shape[0] - crop_size))
            # crop_y = random.randint(0, int(img.shape[1] - crop_size))
            crop_x = int((img.shape[0] - crop_size) / 2)
            crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]
        img_datas.append(img)
    return img_datas

# def read_clip_and_label(batch_size, lines, start_pos=0, num_frames_per_clip=64, sample_rate=1, 
#                         crop_size=224, shuffle=True, add_flow=False, add_rgb=False):
#     read_dirnames = []
#     rgb_data = []
#     flow_data = []
#     label = []
#     batch_index = 0
#     next_batch_start = -1

#     for index in range(len(lines)):
#         if batch_index >= batch_size:
#             next_batch_start = index
#             break
#         line = lines[index].strip('\n').split()
#         dirname = line[0]
#         tmp_label = line[1]
#         if add_rgb:
#             rgb_img_datas = np.load(dirname + '/rgb.npy')
#             rgb_data.append(rgb_img_datas)
#         if add_flow:
#             flow_img_datas = np.load(dirname + '/flow.npy')
#             flow_data.append(flow_img_datas)
#         label.append(int(tmp_label))

#     np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
#     np_arr_flow_data = np.array(flow_data).astype(np.float32)
#     np_arr_label = np.array(label).astype(np.int64)

#     return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size)

def read_clip_and_label(batch_size,  lines, start_pos=0, num_frames_per_clip=64, sample_rate=1, 
                        crop_size=224, add_flow=False, add_rgb=False):
    read_dirnames = []
    rgb_data = []
    flow_data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    for index in range(len(lines)):
        if batch_index >= batch_size:
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        tmp_rgb_data, tmp_flow_data, _= get_frames_data(dirname, num_frames_per_clip, sample_rate, add_flow, add_rgb)
        if add_rgb:
            rgb_img_datas = data_process(tmp_rgb_data, crop_size)
            rgb_data.append(rgb_img_datas)
        if add_flow:
            flow_img_datas = data_process(tmp_flow_data, crop_size)
            flow_data.append(flow_img_datas)
        label.append(int(tmp_label))
        batch_index = batch_index + 1
        read_dirnames.append(dirname)
        # np.save(dirname + '/rgb.npy', rgb_img_datas)
        # np.save(dirname + '/flow.npy', flow_img_datas)

    # pad (duplicate) data/label if less than batch_size
    # valid_len = len(rgb_data)
    # pad_len = batch_size - valid_len
    # if pad_len:
    #     print("pad data!")
    #     for i in range(pad_len):
    #         rgb_data.append(rgb_data[-1])
    #         flow_data.append(flow_data[-1])
    #         label.append(int(label[-1]))

    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size)

# def read_clip_and_label(batch_size, files, start_pos=0, num_frames_per_clip=64, sample_rate=1, 
#                         crop_size=224, shuffle=True, add_flow=False):

#     batch_index = 0
#     next_batch_start = -1
#     read_dirnames = []
#     rgb_data = []
#     flow_data = []
#     label = []
#     for offset in range(0, len(files), batch_size):
#         lines = files[offset:offset+batch_size]
#         for index in range(len(lines)):
#             line = lines[index].strip('\n').split()
#             print(line)
#             dirname = line[0]
#             tmp_label = line[1]

#             tmp_rgb_data, tmp_flow_data, _ = get_frames_data(dirname, num_frames_per_clip, sample_rate, add_flow)
#             if len(tmp_rgb_data) != 0:
#                 rgb_img_datas = data_process(tmp_rgb_data, crop_size)
#                 if add_flow:
#                     flow_img_datas = data_process(tmp_flow_data, crop_size)
#                     flow_data.append(flow_img_datas)
#                 rgb_data.append(rgb_img_datas)
#                 label.append(int(tmp_label))
#                 batch_index = batch_index + 1
#                 read_dirnames.append(dirname)
#         # pad (duplicate) data/label if less than batch_size
#         # valid_len = len(rgb_data)
#         # pad_len = batch_size - valid_len
#         # if pad_len:
#         #     print("pad data!")
#         #     for i in range(pad_len):
#         #         rgb_data.append(rgb_data[-1])
#         #         flow_data.append(flow_data[-1])
#         #         label.append(int(label[-1]))
    
#         np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
#         np_arr_flow_data = np.array(flow_data).astype(np.float32)
#         np_arr_label = np.array(label).astype(np.int64)
#         read_dirnames = []
#         rgb_data = []
#         flow_data = []
#         label = []
        
        
#         yield np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size), next_batch_start, read_dirnames
