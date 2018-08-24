#!/usr/bin/python

"""
A script to delet subfolders and contents when only image are saved without grasp label.
Author: Wenjie Duan
"""

import os
import shutil
import argparse


def cleanFolder(directory):
    cleanedNum = 0
    for dirs, folders,files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(".png") & ("executed.txt" not in files):
                cleanedNum += 1
                print ("BLANK FOLDER:"+dirs)
                shutil.rmtree(dirs)
    print ("Total cleaned folder number:",cleanedNum)

if __name__=='__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Clean the folder of collected data (remove folder without complete data filei.)')
    parser.add_argument('--dir', type=str, default=None, help='the directory which stores the data of one grasp trial. (Example: "~/collection/201808015")')
    args = parser.parse_args()
    directory = args.dir

    # clean all subfolders in the directory
    cleanFolder(directory)
