"""
Normal estimator for noisy point cloud data
Author: Wenjie Duan
"""

import numpy as np
import cv2

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from gqcnn import cv2Visualizer as cv2vis
from . import SuctionPoint2D

from perception import NormalCloudImage,PointCloudImage


class NormalEstimator():
    @staticmethod
    def normal_cloud_im(point_cloud_im):
        """
        get normals of point cloud

        Parameters
        ---
        point_cloud_im: numpy.array
           the point cloud array with size (height,width,3)

        Returns
        ---
        numpy.array
           the normal of input point cloud, with the same size
        """
        data =  point_cloud_im.copy()
        gx,gy,_ = np.gradient(data)
        gx_data = gx.reshape(data.shape[0] * data.shape[1],3) 
        gy_data = gy.reshape(data.shape[0] * data.shape[1],3)
        pc_grads = np.cross(gx_data,gy_data)
        pc_grad_norms = np.linalg.norm(pc_grads, axis=1)
        pc_grads[pc_grad_norms > 0] = pc_grads[pc_grad_norms > 0] / np.tile(pc_grad_norms[pc_grad_norms > 0, np.newaxis], [1,3])
        normal_im_data = pc_grads.reshape(data.shape[0],data.shape[1], 3)
        return normal_im_data 

    @staticmethod
    def smooth(normal_cloud_im,method='average',kernel_size=7):
        """smooth normal cloud image

        Parameters
        ---
        normal_cloud_im: obj:'NormalCloudImage'
           the normals need to be smoothed, size: (height, width, 3)
        method: str
           the method of smoothing normal cloud image. Options are: average, median, gaussian
        kernel_size: int
           the size of the smoothing kernel

        Returns
        ---
        obj:'NormalCloudImage'
           the smoothed normal cloud.
        """
        data = normal_cloud_im.data.copy()
        # convert data to int 0~255
        data += 1
        data /= 2.0
        cvted = np.uint8(np.multiply(data,255))
        # process calculation
        if method == "average":
            processed = cv2.blur(cvted,(kernel_size,kernel_size))
        elif method =="median":
            processed = cv2.medianBlur(cvted,kernel_size)
        elif method == "gaussian":
            processed = cv2.GaussianBlur(cvted,(kernel_size,kernel_size),0)
        # scale back to -1.0~1.0
        processed = np.multiply(processed,1/127.0)
        processed -=1
        pc_grads = processed.reshape(data.shape[0]*data.shape[1], 3)
        pc_grad_norms = np.linalg.norm(pc_grads, axis=1)
        pc_grads[pc_grad_norms > 0] = pc_grads[pc_grad_norms > 0] / np.tile(pc_grad_norms[pc_grad_norms > 0, np.newaxis], [1, 3])
        normal_im_data = pc_grads.reshape(data.shape[0], data.shape[1], 3)

        return NormalCloudImage(normal_im_data,normal_cloud_im.frame)#normal_im_data 

    @staticmethod
    def PCA_grasp(point_cloud_im, grasp, radius=0.005):
        """correct the grasp axis by calculating the PCA of th point cloud

        Parameters
        ---
        point_cloud_im: obj:'PointCloudImage'
           the point cloud of the environment
        grasp: obj: 'SuctionPoint2D'
           the grasp that needs to be corrected
        radius: float
           the search radius in the point cloud, in mm

        Returns
        ---
        obj:'SuctionPoint2D'
           the grasp pose with corrected approaching axis

        """
        # get the point cloud data
        data = point_cloud_im.data.copy()
        # get the grasp center position
        cy=int(grasp.center.y)
        cx=int(grasp.center.x)
        # search neightbors of the grasp point
        neigh = NearestNeighbors(radius=radius, algorithm='kd_tree')
        sample=data[cy-30:cy+30, cx-30:cx+30].reshape(3600,3)
        neigh.fit(sample)
        rng = neigh.radius_neighbors([data[cy,cx]],return_distance=False)
        # Principle Components Analysis based on the neighboring points
        pca = PCA(3)
        pca.fit(sample[rng[0]])
        # return corrected grasp pose with z axis pointing downward
        if pca.components_[2][2]>0:
            grasp.axis=pca.components_[2]
        else:
            grasp.axis=-pca.components_[2]
        return grasp

"""
import time
pci = NormalEstimator()
#depth_im = np.load('depth_im.npy')
point_cloud_im = np.load('point_cloud_im.npy')

normal_cloud_im =pci.normal_cloud_im(point_cloud_im)

show_vertical_im=1
if show_vertical_im:
    vertical_im = np.zeros(normal_cloud_im.shape)
    for c in range(normal_cloud_im.shape[0]):
        for v in range(normal_cloud_im.shape[1]):
            if np.array_equal(normal_cloud_im[c][v],[0,0,-1.0]):
                vertical_im[c][v]=[255,255,255]
    cv2.imshow('vertical_points',vertical_im)
    cv2.waitKey(0)

print("normal_cloud_im max:"+str(np.amax(normal_cloud_im))+"min:"+str(np.amin(normal_cloud_im)))
print("normal_cloud_im shape:"+str(normal_cloud_im.shape))
normal_cloud_im_tmp = cv2vis.normalize(normal_cloud_im,reverse=False)
cv2.imshow('normal_cloud_im',normal_cloud_im_tmp)
cv2.waitKey(0)

start= time.time()
median = pci.smooth(normal_cloud_im,method='average')
end = time.time()
print("normal processing takes:"+str(end-start)+" s")
print("median max:"+str(np.amax(median)))
median_tmp=cv2vis.normalize(median,reverse=False)
cv2.imshow('median_normal_cloud_im',median_tmp)
cv2.waitKey(0)

show_processed_vertical_im=1
if show_processed_vertical_im:
    vertical_im = np.zeros(normal_cloud_im.shape)
    for c in range(normal_cloud_im.shape[0]):
        for v in range(normal_cloud_im.shape[1]):
            if np.array_equal(median[c][v],[0,0,-1.0]):
                vertical_im[c][v]=[255,255,255]
    cv2.imshow('processed_vertical points',vertical_im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import time
cx = 200
cy = 500
point_cloud_im = np.load('point_cloud_im.npy')
point_cloud_im_ori = point_cloud_im.copy()
point_cloud_im = point_cloud_im.reshape(point_cloud_im.shape[0]*point_cloud_im.shape[1],3)
neigh = NearestNeighbors(radius=0.012, algorithm='kd_tree')
print('fitting data...')
start = time.time()
sample=point_cloud_im_ori[cx-30:cx+30,cy-30:cy+30].reshape(3600,3)
neigh.fit(sample)
print("fitting data takes:"+str(time.time()-start))
print('finding eightbors ...')
start = time.time()
rngs = neigh.radius_neighbors([point_cloud_im_ori[cx,cy]],return_distance=False)
print("finding neighbors takes:"+str(time.time()-start))
print("number of neighbors:"+str(rngs[0].shape))
pca = PCA(3)
pca.fit(sample[rngs[0]])
print(pca.components_)
"""

