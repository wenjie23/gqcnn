#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS Subscriber for recording the experiments.
Author: Wenjie Duan
"""
import os
import rospy
import json
import sys
import select
import time
import shutil
import argparse
import signal

import numpy as np

from std_msgs.msg import String
from multiprocessing import Process
from collections import deque

class ExperimentRecorder(object):
    """ Record experiments results """
    # colors for printing text with different color
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, model_dir, output_dir):
        """
        Parameter:
        ---------
        model_dir: str
            path of the model that is used in grasp experiment
        output_dir: str
            directory to which the expeirments will be solved
        """
        self.model_dir = model_dir
        self.output_dir = os.path.join(output_dir,time.strftime('%Y%m%d%H%M%S'))
        # total number of grasps
        self.counter = 0
        # number of success grasps
        self.success_counter = 0
        # a que to stack heared data
        self.dq = deque([None])
        # list to save predictions proposed by model
        self.predictions = []
        # lise to save the grasp result
        self.labels = []
        print("Start experiment\n "+self.output_dir)

    @property
    def success_rate(self):
        """
        Returns
        --------
        the grasp success rate so far
        """
        return self.success_counter/float(self.counter)*100

    def modifyLabel(self, directory):
        """
        Parameters
        ----------
        directory: str
            the path to the collected data trial
        """
        if not self.checkExecution(directory):
            return
        # read label file
        label_path = os.path.join(directory,'label.json')
        with open(label_path) as jsonFile:
            try:
                label_data = json.load(jsonFile)
            except json.decoder.JSONDecodeError:
                print("No valide value now ...")
        if (label_data['success']==1):
            text_color=self.OKGREEN
        else:
            text_color=self.FAIL
        # give user time to modify data
        timeout=5  # seconds left for user to modify data
        print ("You have ",timeout,"seconds to modify the data:") 
        print("[d]: delete entire trial")
        print("[1]: modify to 1")
        print("[0]: modify to 0")
        print(label_path + text_color + " Success Label:" + str(label_data['success']) + self.ENDC)
        delete_data = False
        print("input: ")
        i, o, e = select.select( [sys.stdin], [], [], timeout)
        # modify data if user typed input
        if (i):
            usr_input = sys.stdin.readline().strip()
            if usr_input =='1' or usr_input=='0':
                print (self.WARNING + "Changing label to:",usr_input + self.ENDC)
                label_data['success']=int(usr_input)
                with open(label_path,'w') as jsonFile:
                    json.dump(label_data,jsonFile)
            elif usr_input=='d': 
                delete_data = True
                print (self.WARNING + "Deleting trial data:" + directory + self.ENDC)
                shutil.rmtree(directory)
            else:
                print (self.WARNING + "Illegal input." + self.ENDC)
        else:
            print("['success'] is unchanged")
        # increase the counter and append the latest result into list
        if not delete_data:
            self.predictions.append(label_data['q_value'])
            self.labels.append(label_data['success'])
            print(self.OKBLUE + 'predicted q value:' + str(label_data['q_value']) + self.ENDC)
            self.counter += 1
            if label_data['success']==1:
                self.success_counter +=1

    def checkExecution(self, directory):
        """ Check whether the trial data is executed by the robot, otherwise delete this trial. 
        
        Parameters
        ---------
        directory: str
            the path of the grasp trial folder to be checked
        """

        for dirs, folders,files in os.walk(directory):
            for fileName in files:
                if fileName.endswith(".png") & ("executed.txt" not in files):
                    print (self.WARNING + directory + " is not executed. Deleting data of this trial..." + self.ENDC)
                    shutil.rmtree(dirs)
                    return False
                else:
                    return True

    def printSR(self):
        """ Print the success rate so far. """
        try:
            print (self.OKBLUE + "Success rate: %.2f%% (%d/%d)" % (self.success_rate, self.success_counter, self.counter) + self.ENDC)
        except ZeroDivisionError:
            print (self.OKBLUE + "Success rate: 0.0%% (0/0)" + self.ENDC)
            
    def saveTrial(self):
        """ Save the experiment data. """
        print("Saving experments data to:", self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # save prediction data to .npy data file
        predictions = np.array(self.predictions)
        np.save(os.path.join(self.output_dir,'predictions.npy'), predictions)
        # save label data to .npy data file
        labels = np.array(self.labels)
        np.save(os.path.join(self.output_dir,'labels.npy'), labels)

    def copyModel(self):
        """ Copy the model to the ouput directory for backup """
        print("Coping the used model to:", self.output_dir)
        try:
            shutil.copytree(self.model_dir, os.path.join(self.output_dir, os.path.basename(self.model_dir)))
        # Directories are the same
        except shutil.Error as e:
            print('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('Directory not copied. Error: %s' % e)

    def handler(self,signum, frame):
        """ A handler to end the experiment recorder and save data. """
        print("Caught Ctrl + c, terminating the experiment recorder")
        self.saveTrial()
        self.copyModel()
        exit(0)


def callback(data, recorder):
    """
    callback function when new message is received.
    
    Parameters
    recorder: obj: ExperimentRecorder
        the object of ExperimentRecorder
    """
    directory = data.data
    recorder.dq.appendleft(directory)
    last_dir = recorder.dq.pop()
    if last_dir == None:
        print("No label path from last trial.")
    else:
        recorder.modifyLabel(last_dir)
    # calculate and print success rate
    recorder.printSR()
    print("~~~~~~~~~~~~~~~~~~")

def experiment_recorder(recorder):
    # initialize node
    print("Initializing ROS node.")
    rospy.init_node('experiment_recorder', anonymous = True)
    rospy.Subscriber('latest_col_dir', String, callback, recorder, queue_size=10)
    # run handler to catch terminate keyboard interuption
    signal.signal(signal.SIGINT, recorder.handler)
    rospy.spin()


if __name__=='__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Record grasp experiment results.')    
    parser.add_argument('--model_dir', type=str, default=None, help='path to the model which is used in ths experiment session.')
    parser.add_argument('--output_dir', type=str, default='/home/wduan/data_external/experiments', help='path to the model which is used in ths experiment session.')
    args = parser.parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir
    recorder = ExperimentRecorder(model_dir, output_dir)
    # run experiment recorder
    experiment_recorder(recorder)
    
