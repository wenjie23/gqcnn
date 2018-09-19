"""
Grasping data recorder
Author: Wenjie Duan
"""

import os
import rospy
import json
import numpy as np

from perception import DepthImage

from . import SuctionPoint2D,GQCNN

class DataRecorder(object):
    """Recorder to save grasping policy data
    """
    def __init__(self,config,save_dir):
        self.config = config
        self.save_dir = save_dir

        # init GQ-CNN
        self._gqcnn = GQCNN.load(self.config['metric']['gqcnn_model'])

    @property
    def gqcnn(self):
        return self._gqcnn

    def save(self, depth_im, grasp):
        """
        Parameters
          depth_im:  :obj:`perception.DepthImage`
            DepthImage from perception module, in which individual pixels have a depth channel 
          grasp: :obj:`GraspAction`
            Executed grasp
        """
        # save depth image to npy data
        if self.config['collection']['depth_image']:
            depth_im_path = os.path.join(self.save_dir,"depth.npy")
            rospy.loginfo("Saving depth image to %s" % depth_im_path)
            depth_im.save(depth_im_path)

        # save tf image to npy data with original resolution
        if self.config['collection']['ori_tf_image']:
            ori_tf_image = self._ori_tf_image(depth_im,grasp)
            ori_tf_image_path = os.path.join(self.save_dir,"ori_tf_image.npy")
            rospy.loginfo("Saving original tf image to %s" % ori_tf_image_path)
            np.save(ori_tf_image_path,ori_tf_image)

        # save the cropped and resaled tf image
        if self.config['collection']['tf_image']:
            tf_image = self._tf_image(depth_im,grasp)
            tf_image_path = os.path.join(self.save_dir,"tf_image.npy")
            rospy.loginfo("Saving tf image to %s " % tf_image_path)
            np.save(tf_image_path,tf_image)

        # save executed grasp poses
        if self.config['collection']['grasp_pose']:
            pose_tensor = self._pose_tensor(depth_im,grasp)
            pose_tensor_path = os.path.join(self.save_dir,"pose_tensor.npy")
            rospy.loginfo("Saving grasp pose to %s " % pose_tensor_path)
            np.save(pose_tensor_path,pose_tensor)

        # save the grasp label
        if self.config['collection']['label']:
            label_path=os.path.join(self.save_dir,"label.json")
            data={"q_value": grasp.q_value,
                    "success": 0} # set success label as 0, will changed latere by COMMAND SERVER 
            rospy.loginfo("Saving label to %s " % label_path)
            with open(label_path,'w')  as jsonFile:
                json.dump(data,jsonFile)

    def _ori_tf_image(self, depth_im, grasp):
        """
        Get the tf_image of the grasp with origianl resolution
        """
        translation = np.array([depth_im.center[0] - grasp.grasp.center.data[1],
                                depth_im.center[1] - grasp.grasp.center.data[0]])
        im_tf = depth_im.transform(translation,grasp.grasp.angle)
        im_tf = im_tf.crop(self.config['metric']['crop_height'],self.config['metric']['crop_width'])
        image_tensor = im_tf.raw_data
        return image_tensor
    
    def _tf_image(self, depth_im, grasp):
        """
        Get the tf_image of the grasp with gqcnn resolution
        """
        scale = float(self.gqcnn.im_height)/self.config['metric']['crop_height']
        depth_im_scaled = depth_im.resize(scale)
        translation = scale * np.array([depth_im.center[0] - grasp.grasp.center.data[1],
                                        depth_im.center[1] - grasp.grasp.center.data[0]])
        im_tf = depth_im_scaled.transform(translation,grasp.grasp.angle)
        im_tf = im_tf.crop(self.gqcnn.im_height,self.gqcnn.im_width)
        image_tensor = im_tf.raw_data
        return image_tensor

    def _pose_tensor(self,depth_im,grasp):
        # extract values to form a numpy array: from first to last ones are: center posidtion index (x,y), grasp depth, grasp position realtive to camera(x,y,z)
        # grasp apprach axis(u,v,w) relative to camera, apprach angle relative to camera optical axis, suction cup diameter in mm.
        pose_tensor = np.zeros(11)
        pose_tensor[0] = grasp.grasp.center.x
        pose_tensor[1] = grasp.grasp.center.y
        pose_tensor[2] = grasp.grasp.depth
        pose_tensor[3:6] = grasp.grasp.pose().translation
        pose_tensor[6:9] = grasp.grasp.axis
        pose_tensor[9] = grasp.grasp.approach_angle
        pose_tensor[10] = self.config['collection']['suction_cup_diameter']
        return pose_tensor

