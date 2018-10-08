# -*- coding: utf-8 -*-
"""
Analyze experiments data.

Author
------
Wenjie Duan
"""

import argparse
import logging
import os
import time
import os
import json
import shutil

from gqcnn import GQCNNAnalyzer
from autolab_core import BinaryClassificationResult

import numpy as np 

def analyze_experiments(input_dirs, output_dir = None):
    """Analyze experiments data and save analyzed data
    Parameters
    ----------
    input_dirs: list
        list of strings whicht are paths to experiments raw data.
    output_dir: str
        path to save the analyzed data.
    """


    models = []
    predictions = []
    labels = []
    if len(input_dirs)>1:
        print(len(input_dirs),"are loaded, their data will be emergered together to ", output_dir)
    for input_dir in input_dirs:
        for dirs, folders, files in os.walk(input_dir):
            print(dirs)
            print(folders)
            print(files)
            models.append(folders)
            predictions.append(np.load(os.path.join(dirs, 'predictions.npy')))
            labels.append(np.load(os.path.join(dirs,'labels.npy')))
            break
    # concatenate arrays
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    for i in range(labels.shape[0]):
        print(str(i),':', labels[i])
    # create folder
    if not (all(x==models[0] for x in models)):
        raise Exception("The loaded experiments uses different models!")

    exp_result = BinaryClassificationResult(predictions, labels)
    print('Grasp number(success/total): ' + str(exp_result.num_true_pos) + '/' + str(exp_result.num_datapoints))
    success_rate = exp_result.num_true_pos/float(exp_result.num_datapoints)
    print('Success Rate:', success_rate)
    average_precision_score = exp_result.ap_score
    accuracy = exp_result.accuracy
    num_objects = int(input("Number of objects (be used to calculate the Percent Cleared):\n"))
    percent_cleared = float(exp_result.num_true_pos)/num_objects
    num_one_time = int(input("Number of objects (be used to calculate the one-time success_rate:)\n"))
    one_time_success_rate = num_one_time/float(num_objects)
    description = input("Research's description about this experiment session:")

    # save experiments stats
    exp_stats ={
            'description': description,
            'success_rate': success_rate,
            'average_precision_score': average_precision_score,
            'accuracy': accuracy,
            'percent_cleared': percent_cleared,
            'one_time_success_rate': one_time_success_rate,
            'success_grasps': int(exp_result.num_true_pos),
            'num_grasps': int(exp_result.num_datapoints),
            'num_objects': num_objects,
            'num_one_time': num_one_time
    }
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('experiment summary:')
    print(exp_stats)
    exp_stats_filename = os.path.join(output_dir, 'exp_stats.json')
    json.dump(exp_stats, open(exp_stats_filename, 'w'),
            indent=2,
            sort_keys=False)
    # save numpy file
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    # copy models
    print(input_dirs[0])
    print(models[0][0])
    try:
        shutil.copytree(os.path.join(input_dirs[0], models[0][0]), os.path.join(output_dir, models[0][0]))
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Analyze the experiments data.')
    parser.add_argument('--input_dirs', '--nargs', nargs='+', help='list of directories to the saved expeirments data.')
    parser.add_argument('--output_dir', type=str, default=None, help='the directory to save the analyzed data.')
    args = parser.parse_args()
    input_dirs = args.input_dirs
    output_dir = args.output_dir
    # analyze experiments data
    analyze_experiments(input_dirs, output_dir)
