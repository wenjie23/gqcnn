# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Analyzes a GQ-CNN model.

Author
------
Vishal Satish and Jeff Mahler
"""
import argparse
import logging
import os
import time
import os

from autolab_core import YamlConfig
from gqcnn import GQCNNAnalyzer

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Analyze a Grasp Quality Convolutional Neural Network with TensorFlow')
    parser.add_argument('model_dir', type=str, default=None, help='path to the model to analyze')
    parser.add_argument('--output_dir', type=str, default=None, help='path to save the analysis')
    parser.add_argument('--dataset_config_filename', type=str, default=None, help='path to a configuration file for testing on a custom dataset')
    parser.add_argument('--config_filename', type=str, default=None, help='path to the configuration file to use')
    parser.add_argument('--splits_dir', type=str, default=None, help='path to load pkl files of train & val indices')
    args = parser.parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir
    dataset_config_filename = args.dataset_config_filename
    config_filename = args.config_filename
    splits_dir = args.splits_dir

    # set defaults
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  '../analysis')
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/analyze_gqcnn_performance.yaml')

    # turn relative paths absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)
    if dataset_config_filename is not None and not os.path.isabs(dataset_config_filename):
        config_filename = os.path.join(os.getcwd(), dataset_config_filename)

    # make the output dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # read config
    config = YamlConfig(config_filename)

    dataset_config = None
    if dataset_config_filename is not None:
        dataset_config = YamlConfig(dataset_config_filename)
    
    # run the analyzer
    analyzer = GQCNNAnalyzer(config)
    analyzer.analyze(model_dir, output_dir, dataset_config, splits_dir)
