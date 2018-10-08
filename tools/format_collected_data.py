# -*- coding: utf-8 -*-
"""
A tool to format data collected on real robot to data format that can be used for training.
Author: Wenjie Duan
"""

import os
import sys
import argparse
import json

import numpy as np

NUM_OF_DATAPOINT = 1000 # number of data points saved in each file

# the file names of auto saved data
TF_IMAGE_FILENAME = "tf_image.npy"
ORI_TF_IMAGE_FILENAME = "ori_tf_image.npy"
POSE_FILENAME = "pose_tensor.npy"
LABEL_FILENAME = "label.json"

# the prefix of npz files that will be saved
TF_OUTPUT_FILENAME_PRE = "depth_ims_tf_table"
ORI_TF_OUTPUT_FILENAME_PRE = "depth_ims_ori_tf_table"
POSE_OUTPUT_FILENAME_PRE = "hand_poses"
LABEL_OUTPUT_FILENAME_PRE = "real_robot_graspability"

def save_data(output_dir, output_file_pre, output_file, file_index):
    output_filename = output_file_pre + '_' + '%05d'%file_index + '.npz'
    print ("Saving file:",output_filename)
    print("Shape:",output_file.shape)
    output_path = os.path.join(output_dir, output_filename)
    np.savez(output_path,output_file)

def load_data(input_dir,output_dir):
    # load data in the directory
    print("Loading data from:",input_dir)
    # initializing numpy arrays
    tf_image = np.zeros((NUM_OF_DATAPOINT, 32, 32, 1))
    ori_tf_image = np.zeros((NUM_OF_DATAPOINT,155,155,1))
    pose_data = np.zeros((NUM_OF_DATAPOINT,7))
    label_data = np.zeros((NUM_OF_DATAPOINT))
    tf_image_num = 0
    tf_file_index = 0
    ori_tf_image_num = 0
    ori_tf_file_index = 0
    pose_num = 0
    pose_file_index = 0
    label_num = 0
    label_file_index = 0

    # scan input directories
    for dirs, folders, files in os.walk(input_dir):

        if TF_IMAGE_FILENAME in files:
            tf_image_arr = np.load(os.path.join(dirs, TF_IMAGE_FILENAME))
            tf_image[tf_image_num] = tf_image_arr
            tf_image_num += 1
            # check number of data points saved in a numpy array
            if tf_image_num >= NUM_OF_DATAPOINT:
                # save data
                save_data(output_dir, TF_OUTPUT_FILENAME_PRE, tf_image, tf_file_index)
                tf_file_index+=1
                tf_image_num = 0


        if ORI_TF_IMAGE_FILENAME in files:
            ori_tf_image_arr = np.load(os.path.join(dirs, ORI_TF_IMAGE_FILENAME))
            ori_tf_image[ori_tf_image_num] = ori_tf_image_arr
            ori_tf_image_num += 1
            # check number of data points saved in a numpy array
            if ori_tf_image_num >= NUM_OF_DATAPOINT:
                # save data
                save_data(output_dir, ORI_TF_OUTPUT_FILENAME_PRE, ori_tf_image, ori_tf_file_index)
                ori_tf_file_index+=1
                ori_tf_image_num = 0
        
        if POSE_FILENAME in files:
            pose_arr  = np.load(os.path.join(dirs, POSE_FILENAME))
            # read sub-elements of the pose array
            pose_data[pose_num] = np.r_[pose_arr[0:3], pose_arr[9:10], pose_arr[3:6]]
            pose_num += 1
            # check number of data points saved in a numpy array
            if pose_num >= NUM_OF_DATAPOINT:
                # save data
                save_data(output_dir, POSE_OUTPUT_FILENAME_PRE , pose_data, pose_file_index)
                pose_file_index += 1
                pose_num  = 0

        if LABEL_FILENAME in files:
            with open(os.path.join(dirs, LABEL_FILENAME)) as jsonFile:
                try:
                    label_json = json.load(jsonFile)
                except json.decoder.JSONDecodeError:
                    print("No valide value now ...")
            label_data[label_num] = label_json['success']
            label_num += 1
            # check number of data points saved in a numpy array
            if label_num >= NUM_OF_DATAPOINT:
                # save data
                save_data(output_dir, LABEL_OUTPUT_FILENAME_PRE , label_data, label_file_index)
                label_file_index += 1
                label_num  = 0

    # cut out unused space
    tf_image = tf_image[0:tf_image_num]
    ori_tf_image = ori_tf_image[0:ori_tf_image_num]
    pose_data = pose_data[0:pose_num]
    label_data = label_data[0:label_num]
    
    # save last files
    save_data(output_dir, TF_OUTPUT_FILENAME_PRE, tf_image, tf_file_index)
    save_data(output_dir, ORI_TF_OUTPUT_FILENAME_PRE, ori_tf_image, ori_tf_file_index)
    save_data(output_dir, POSE_OUTPUT_FILENAME_PRE, pose_data, pose_file_index)
    save_data(output_dir, LABEL_OUTPUT_FILENAME_PRE , label_data, label_file_index)

    ## check loaded data
    #tf_image_num = 0
    #for dirs, folders, files in os.walk(input_dir):
    #    if TF_IMAGE_FILENAME in files:
    #        tf_image_arr = np.load(os.path.join(dirs, TF_IMAGE_FILENAME))
    #        if np.array_equal(tf_image[tf_image_num], tf_image_arr):
    #            print("!!!!")
    #        tf_image_num += 1


if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Format collected data.')
    parser.add_argument('--input_dir', type=str, default=None, help='the directory which stores the data of collected data.')
    parser.add_argument('--output_dir', type=str, default=None, help='the directory to save the formatted data.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # load input datas
    load_data(input_dir,output_dir)
