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
Script to plot the various errors saved during training.

Author
------
Jeff Mahler

Required Parameters
------------------------
model_dir : str
    Command line argument, the path to the model whose errors are to plotted. All plots and other metrics will
    be saved to this directory. 
"""
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

TRAIN_LOSS_FILENAME = 'train_losses.npy'
TRAIN_ERRORS_FILENAME = 'train_errors.npy'
VAL_ERRORS_FILENAME = 'val_errors.npy'
TRAIN_ITERS_FILENAME = 'train_eval_iters.npy'
VAL_ITERS_FILENAME = 'val_eval_iters.npy'


if __name__ == '__main__':
    result_dir = sys.argv[1]
    train_errors_filename = os.path.join(result_dir, TRAIN_ERRORS_FILENAME)
    val_errors_filename = os.path.join(result_dir, VAL_ERRORS_FILENAME)
    train_iters_filename = os.path.join(result_dir, TRAIN_ITERS_FILENAME)
    val_iters_filename = os.path.join(result_dir, VAL_ITERS_FILENAME)

    train_errors = np.load(train_errors_filename)
    val_errors = np.load(val_errors_filename)
    train_iters = np.load(train_iters_filename)
    val_iters = np.load(val_iters_filename)

    init_val_error = val_errors[0]
    norm_train_errors = train_errors / init_val_error
    norm_val_errors = val_errors / init_val_error

    print 'Final val error', val_errors[-1]

    plt.figure()
    plt.plot(train_iters, train_errors, linewidth=4, color='b')
    plt.plot(val_iters, val_errors, linewidth=4, color='g')
    plt.ylim(0, 100)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Error Rate', fontsize=15)
 
    plt.figure()
    plt.plot(train_iters, norm_train_errors, linewidth=4, color='b')
    plt.plot(val_iters, norm_val_errors, linewidth=4, color='g')
    plt.ylim(0, 2.0)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Error Rate', fontsize=15)
    plt.show()
 
    plt.figure(figsize=(8,6))
    plt.plot(train_iters, train_errors, linewidth=4, color='b')
    plt.plot(val_iters, val_errors, linewidth=4, color='g')
    plt.ylim(0, 100)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Error Rate', fontsize=15)
    plt.savefig(os.path.join(result_dir, 'training_curve.jpg'))
 
    plt.figure(figsize=(8,6))
    plt.plot(train_iters, norm_train_errors, linewidth=4, color='b')
    plt.plot(val_iters, norm_val_errors, linewidth=4, color='g')
    plt.ylim(0, 2.0)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Error Rate', fontsize=15)
    plt.savefig(os.path.join(result_dir, 'normalized_training_curve.jpg'))
  
