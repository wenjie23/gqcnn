# -*- coding: utf-8 -*-
"""
A tool to format data collected on real robot to data format that can be used for training.
Author: Wenjie Duan
"""

import os
import sys
import argparse

import numpy as np

NUM_OF_DATAPOINT = 4 # number of data points saved in each file
TF_IMAGE_FILENAME = "tf_image.npy"
ORI_TF_IMAGE_FILENAME = "ori_tf_image.npy"

def load_data(input_dir):
    # load data in the directory
    print("Loading data from:",input_dir)
    # initializing numpy arrays
    tf_image = np.zeros((NUM_OF_DATAPOINT, 32, 32, 1))
    print(tf_image.shape)
    tf_image_num = 0

    # scan input directories
    for dirs, folders, files in os.walk(input_dir):
        if tf_image_num > NUM_OF_DATAPOINT:
            print("oh no")
            print(tf_image.shape)

            # save data
            TODO

        if TF_IMAGE_FILENAME in files:
            tf_image_arr = np.load(os.path.join(dirs, TF_IMAGE_FILENAME))
            tf_image[tf_image_num] = tf_image_arr
            tf_image_num += 1

    # cut out unused space
    tf_image = tf_image[0:tf_image_num]
    print(tf_image.shape)

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
    load_data(input_dir)
