#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS Subscriber for monitoring the latest collected data
Author: Wenjie Duan
"""

import os
import rospy
import json
import sys
import select
import time
import shutil
import numpy as np

from std_msgs.msg import String
from multiprocessing import Process
from collections import deque

# global variables
counter = 0
success_counter = 0
dq = deque([None])

# colors for printing text with different color
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printSR():
    global counter
    global success_counter
    try:
        success_rate = success_counter/float(counter)*100
        print (bcolors.OKBLUE + "Success rate: %.2f%% (%d/%d)" % (success_rate, success_counter, counter) + bcolors.ENDC)
    except ZeroDivisionError:
        print (bcolors.OKBLUE + "Success rate: 0.0%% (0/0)" + bcolors.ENDC)    


def checkExecution(directory):
    # check whether the trial data is executed by the robot, otherwise delete this trial
    for dirs, folders,files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(".png") & ("executed.txt" not in files):
                print (bcolors.WARNING + directory + " is not executed. Deleting data of this trial..." + bcolors.ENDC)
                shutil.rmtree(dirs)
                return False
            else:
                return True
def scanCollection(directory):
    global counter
    global success_counter
    print("Loading collected data...")
    for dirs, folders, files in os.walk(directory):
        for fileName in files:
            if fileName == "label.json":
                counter +=1
                label_path = os.path.join(dirs,fileName)
                with open(label_path) as jsonFile:
                    try:
                        label_data = json.load(jsonFile)
                    except json.decoder.JSONDecodeError:
                        print("No valide value now ...")
                if (label_data['success']==1):
                    success_counter +=1
    printSR()
    print("~~~~~~~~~~~~~~~~~~")


def modifyLabel(directory):
    if not checkExecution(directory):
        return
    # read label file
    label_path = os.path.join(directory,'label.json')
    with open(label_path) as jsonFile:
        try:
            label_data = json.load(jsonFile)
        except json.decoder.JSONDecodeError:
            print("No valide value now ...")
    if (label_data['success']==1):
        text_color=bcolors.OKGREEN
    else:
        text_color=bcolors.FAIL
    # give user time to modify data
    timeout=5  # seconds left for user to modify data
    print ("You have ",timeout,"seconds to modify the data:") 
    print("[d]: delete entire trial")
    print("[1]: modify to 1")
    print("[0]: modify to 0")
    print(label_path + text_color + " Success Label:" + str(label_data['success']) + bcolors.ENDC)
    delete_data = False
    print("input: ")
    i, o, e = select.select( [sys.stdin], [], [], timeout)
    # modify data if user typed input
    if (i):
        usr_input = sys.stdin.readline().strip()
        if usr_input =='1' or usr_input=='0':
            print (bcolors.WARNING + "Changing label to:",usr_input + bcolors.ENDC)
            label_data['success']=int(usr_input)
            with open(label_path,'w') as jsonFile:
                json.dump(label_data,jsonFile)
        elif usr_input=='d': 
            delete_data = True
            print (bcolors.WARNING + "Deleting trial data:" + directory + bcolors.ENDC)
            shutil.rmtree(directory)
        else:
            print (bcolors.WARNING + "Illegal input." + bcolors.ENDC)
    else:
        print("['success'] is unchanged")
    # increase the counter
    if not delete_data:
        global counter
        counter = counter + 1
        if label_data['success']==1:
            global success_counter
            success_counter +=1
    

def callback(data):
    directory = data.data
    global dq
    dq.appendleft(directory)
    last_dir = dq.pop()
    if last_dir == None:
        print("No label path from last trial.")
    else:
        modifyLabel(last_dir)
    # calculate and print success rate
    printSR()
    print("~~~~~~~~~~~~~~~~~~")


def monitor():
    # initialize node
    rospy.init_node('monitor', anonymous = True)
    rospy.Subscriber('latest_col_dir', String, callback, queue_size=10)
    rospy.spin()

if __name__=='__main__':
    # if start the node with directory input, then the grasp counter will be initialized based on data in the folder
    if len(sys.argv)>1:
        scanCollection(sys.argv[1])
    monitor()
