"""
A tool script to inspect collected data.
Author: Wenjie
"""

import numpy as np
import os
import sys
import cv2
import shutil
import argparse
import json
import pylab

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pprint import pprint
from visualize_depth_image import visualize_depth_image


def visualize_grasp (depth_data,grasp_x,grasp_y,grasp_depth,q_value):
    """
    Visualize depth image in gray color and the corresponding grasp with q value in red-yellow-green color code.
    Parameters
    ---
    depth_data: obj:numpy.array
      the depth image data with shape (height, widht, 1)
    grasp_x: obj: `int` or `float`
      the grasp position row index in the depth image
    grasp_y: obj: `int` or `float`
      the grasp position column index in the depth image 
    grasp_depth: float
      the grasp position depth relative the the camera
    q_value: float
      the q value predicted by GQCNN
    """
    color = plt.cm.RdYlGn(q_value)
    plt.figure(figsize=(16,16))
    plt.imshow(depth_data[:,:,0],cmap=plt.cm.gray_r)
    plt.scatter(grasp_x,grasp_y,c=color,marker='.',s=125.0)
    plt.title("The executed grasp: depth:%.3f q:%.3f" % (grasp_depth,q_value))

def inspect_single_data(directory):
    # scan file names in directory
    fileNames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # print founded files 
    print("Files in the folder:")
    for fileName in fileNames:
        print(fileName)
    # read and display each file
    for fileName in fileNames:
        file_root,file_ext = os.path.splitext(fileName)
        # show .png image
        if file_ext == ".png":
            img = cv2.imread(os.path.join(directory,fileName))
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title(fileName)
            plt.show(block=False)
    for fileName in fileNames:
        file_root,file_ext = os.path.splitext(fileName)
        # print .json file
        if file_ext == ".json":
            print(fileName+" :")
            with open(os.path.join(directory,fileName)) as jsonFile:
                if file_root =="calibration":
                    calibration_data = json.load(jsonFile)
                    pprint(calibration_data)
                elif file_root == "label":
                    label_data = json.load(jsonFile)
                    pprint(label_data)
        # print of visualize .npy file
        elif file_ext == '.npy':
            if file_root =="pose_tensor":
                pose_data = np.load(os.path.join(directory,fileName))
                print(fileName+" :")
                print(pose_data)
            elif file_root == "depth":
                depth_data = np.load(os.path.join(directory,fileName))
                visualize_depth_image(depth_data,fileName)
            elif file_root == "tf_image":
                tf_image_data = np.load(os.path.join(directory,fileName))
                visualize_depth_image(tf_image_data,fileName)
            elif file_root == "ori_tf_image":
                ori_tf_image_data = np.load(os.path.join(directory,fileName))
                visualize_depth_image(ori_tf_image_data,fileName)

    # visualize the grasp the inspected grasp trial
    visualize_grasp(depth_data,pose_data[0],pose_data[1],pose_data[2],label_data["q_value"])

    plt.show()

    usr_input = input("Type in '1' or '0' to change the success label, 'd' to delete this grasp trial.")
    if usr_input =='1' or usr_input=='0':
        print ("Changing label to:",usr_input)
        label_data['success']=int(usr_input)
        with open(os.path.join(directory,'label.json'),'w') as jsonFile:
            json.dump(label_data,jsonFile)
    elif usr_input=='d': 
        print ("Deleting trial data:"+directory)
        shutil.rmtree(directory)
    else:
        print ("Illegal input.")

def inspect_range_data(mul_directory):
    trial_num = 0
    success_num = 0
    q_values = []
    fail_q_values = []
    success_q_values = []
    success_labels = []
    for directory in mul_directory:
        for dirs, folders, files in os.walk(directory):
            for fileName in files:
                if fileName == 'label.json':
                    with open(os.path.join(dirs,fileName)) as jsonFile:
                        label_data = json.load(jsonFile)
                        trial_num+=1
                        q_values.append(label_data['q_value'])
                        success_labels.append(label_data['success'])
                        if label_data['success']==0:
                            fail_q_values.append(label_data['q_value'])
                        else:
                            success_q_values.append(label_data['q_value'])
                            success_num +=1

    print('number of trials:',trial_num)
    print('success number:',success_num)
    y,binEdges = np.histogram(q_values,bins=100) 
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    pylab.plot(bincenters,y,'-',label='tol')
    pylab.title('q values histogram')

    y,binedges = np.histogram(success_q_values,bins=100) 
    bincenters = 0.5*(binedges[1:]+binedges[:-1])
    pylab.plot(bincenters,y,'-g',label='success grasp')
    pylab.title('success q values histogram')

    y,binEdges = np.histogram(fail_q_values,bins=100) 
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    pylab.plot(bincenters,y,'-r', label='failed grasp')
    pylab.legend(loc='upper right')
    pylab.xlabel('q values')
    pylab.ylabel('grasp numbers')
    pylab.title('Grasp Histogram')
    pylab.show()
if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Inspect collected data')
    parser.add_argument('--one', type=str, default=None, help='the directory which stores the data of one grasp trial. (Example: ~/collection/20180801/20180801113645)')
    parser.add_argument('--mul', '--nargs', nargs='+',help='list of directories you want to inspect. (Example: ~/collection/2018080817 ~/collection/20180818)')
    args = parser.parse_args()
    directory = args.one
    mul_directory = args.mul
    
    # inspect a single trial data
    if directory:
        inspect_single_data(directory)
    if mul_directory:
        inspect_range_data(mul_directory)



