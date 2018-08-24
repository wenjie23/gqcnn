"""
Visualizer by using OpenCV
Author: Wenjie Duan
"""
import numpy as np 
import cv2
import colorsys

from gqcnn import Grasp2D,SuctionPoint2D


class cv2Visualizer:
    @staticmethod
    def normalize(image,reverse=True):
        """
        normalize image to range 0~1.

        Parameters
        ---
        image: numpy.array
            the image that needed to be normalized
        reverse: bool
            Whether to rever the image. If true, mximum value is normalized to 0, and minimux value is normalized to 1.

        Returns
        ---
        numpy.array
            the image that have been normalized to 0~1
        """
        array = image.copy()
        if reverse:
            array-=np.amax(array)
            array=-array
        else:
            array-=np.amin(array)
        array/=float(np.amax(array))
        return array
    
    @staticmethod
    def grasp(image,grasp,q=0):
        """
        draw grasp on the image.

        Parameters
        ---
        image: numpy.array
            the image that will be shown.
        grasp: obj:'SuctionPoint2D'
            grasp to plot
        q: float
            float number from 0 to 1. the grasp confidence.

        Returns
        ----
        numpy.array
            the image that has been drawn with grasp on it
        """
        image_grasp = image.copy()
        h = q/3
        s=0.8
        l=0.5
        r,g,b = colorsys.hls_to_rgb(h,l,s)
        color = (b,g,r)
        cv2.circle(image_grasp,(int(grasp.center.x),int(grasp.center.y)),5,color,2)
        return image_grasp 

    @staticmethod
    def imshow(image,title='image'):
        """
        shwo the image by using opencv imshow()
        """
        print('do somthing cv2 imshow')
        cv2.namedWindow(title)
        cv2.imshow(title,image)
        print('cv2 did')
        cv2.waitKey(0)
        print('input some key')
        cv2.destroyWindow(title)
        print('window is destroied')

'''
cv2vis = cv2Visualizer()
depth_im = np.load('depth_im.npy')
depth_im_tmp = cv2vis.normalize(depth_im)
#cv2vis.imshow(depth_im_tmp,'depth_im')
#cv2vis.imshow(depth_im_tmp,'depth_im')
img = cv2.imread('test.png',0)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import os 
os.mkdir('/home/wduan/data_collection')
cv2.imwrite('/home/wduan/data_collection/test2.png',img)
'''
