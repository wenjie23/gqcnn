"""
A simple script to visulize depth image stored in .npy or .npz file

Author
------
Wenjie Duan
"""

import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Visulize depth image in 3D')
    parser.add_argument('--fileName', type=str, default=None, help='path to a depth image stored as a .npy file or .npz file')
    parser.add_argument('--ind', type=int, default=None, help='index of depth image stored in .npz file')
    args = parser.parse_args()
    fileName=args.fileName
    ind=args.ind
    file_root, file_ext = os.path.splitext(fileName)

    # read image file
    data=np.load(fileName)
    if file_ext=='.npy':
        pass
    elif file_ext=='.npz':
        raw_data=data['arr_0']
        data=raw_data[ind]

    # visualize npy file
    fig=plt.figure()
    a=data
    ny=a.shape[0]
    nx=a.shape[1]
    x=np.linspace(0,1,nx)
    y=np.linspace(0,1,ny)
    xv,yv=np.meshgrid(x,y)
    ax=fig.add_subplot(111,projection='3d')
    if len(a.shape)==3:
        a=np.squeeze(a,axis=(2,))
    dem3d=ax.plot_surface(xv,yv,a,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    fig.colorbar(dem3d,shrink=0.5,aspect=5)
    plt.show()