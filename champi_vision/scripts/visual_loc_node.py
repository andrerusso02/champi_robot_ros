#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy.spatial.transform import Rotation as R

import champi_vision.bird_view as bv

from icecream import ic

class VisualLocalizationNode(Node):

    def __init__(self):
        super().__init__('visual_loc')

        self.current_pos = np.array([2.6, 1.7])
        self.current_angle = -1.629

        self.enable_masking = True


        # Parameters
        self.enable_viz_keypoints = self.declare_parameter('enable_viz_keypoints', True).value
        self.enable_match_viz = self.declare_parameter('enable_match_viz', False).value

        # Print parameters
        self.get_logger().info(f"enable_viz_keypoints: {self.enable_viz_keypoints}")
        self.get_logger().info(f"enable_match_viz: {self.enable_match_viz}")


        # handle camera info
        self.camera_info = None
        self.subscription_cam_info = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.callback_cam_info,
            10)


        # transformation matrix between base_link and camera using tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bird_view = None

        # Map to undistort the image
        self.map1 = None
        self.map2 = None

        # other stuff

        self.img_viz = None

        self.pts_mask_viz = None

        self.cv_bridge = CvBridge()

        self.curent_image = None
        self.time_last_image = time.time()
        
        self.pos_cam_in_bird_view_pxls = None

        ref_img_path = get_package_share_directory('champi_vision') + '/ressources/images/ref_img.png'
        self.ref_img = self.load_ref_image(ref_img_path)
        self.pxl_to_m_ref = self.ref_img.shape[1]/3.0

        # detection / matching init
        self.sift_detector = cv2.SIFT_create()
        self.ref_kp, self.ref_des =  self.sift_detector.detectAndCompute(self.ref_img, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)


        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning


    def init_bird_view(self):
        transform = None
        try:
            # get transform, which is what we need to wait for
            transform = self.tf_buffer.lookup_transform('base_link', 'camera',rclpy.time.Time().to_msg()) # todo use camera info

            # # unsubscribe from camera info (can't do that in the callback otherwise the nodes crashes) TODO IT MAKES THE NODE CRASH
            # self.subscription_cam_info.destroy()

            # compute transform between the 2 frames as 1 matrix
            rot = R.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
            rot = rot.as_matrix()
            trans = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
            transform_mtx = np.concatenate([rot, trans.reshape(3,1)], axis=1)
            transform_mtx = np.concatenate([transform_mtx, np.array([[0,0,0,1]])], axis=0)

            # initialize bird view
            K = np.array(self.camera_info.k).reshape(3,3)
            self.bird_view = bv.BirdView(K, transform_mtx, (0.5, -1.1), (2.7, 1.1), resolution=378) 

            self.pos_cam_in_bird_view_pxls = self.bird_view.get_work_plane_pt_in_bird_img(np.array([0, 0, 1]))
            ic(self.pos_cam_in_bird_view_pxls)

            # compute undistortion map
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, np.array(self.camera_info.d), None, K, (640,480), cv2.CV_32FC1)

            self.get_logger().info("Node Initialized", once=True)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info("waiting for tf2 transform...", once=True)
            return
    
    def preprocess(self, image):

        # median blur
        # image = cv2.medianBlur(image, 11)

        pass

    def callback_cam_info(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
    

    def draw_oriented_rectangle(self, mask, pos, angle, size):
        # Convert position from meters to pixels
        pos = pos * self.pxl_to_m_ref

        # Create a rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) 

        # Compute the 4 corners of the rectangle
        corners = np.array([[0.5, 0], [0.5, -1.5], [-0.5, -1.5], [-0.5, 0]]) * self.pxl_to_m_ref
        corners = np.dot(corners, R.T) + pos

        self.pts_mask_viz = corners

        ic(corners)

        # Draw the rectangle
        cv2.fillPoly(mask, [corners.astype(np.int32)], 1)

    

    def make_mask(self):
        """Creates a mask of the same dimension as the ref image,
        with an oriented rectanle of ones where the robot is supposed to be."""

        mask = np.zeros(self.ref_img.shape, dtype=np.uint8)

        self.draw_oriented_rectangle(mask, self.current_pos, self.current_angle, np.array([0.5, 0.5]))

        return mask



    def image_callback(self, msg):

        # get camera info if not already
        if self.camera_info is None:
            self.get_logger().info("waiting for camera info...", once=True)

            return
        
        self.get_logger().info("camera info received", once=True)


        # initialize bird view if not already
        if self.bird_view is None:
            self.init_bird_view()


        # convert to cv2
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # undistort
        cv_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        self.curent_image = cv_image

        # get bird view
        bird_view_img = self.bird_view.project_img_to_bird(self.curent_image)


        # Save bird view to file
        # cv2.imwrite("bird_view.png", bird_view_img)

        self.preprocess(bird_view_img)

        # get homography
        ok, M, matchesMask, kp1, kp2, good = self.get_affine_transform_bird_to_ref(bird_view_img)


        if not ok or M is None:
            self.visualization()
            return
        
        # get scale
        scale = np.linalg.norm(M[:2,0])

        if scale < 0.4 or scale > 0.6:
            self.visualization()
            return


        angle = self.get_angle(M)

        pos_cam_in_ref_pxls = np.matmul(M, self.pos_cam_in_bird_view_pxls)
        pos_cam_in_ref_pxls = pos_cam_in_ref_pxls[:2] / pos_cam_in_ref_pxls[2]
        pos_cam_in_ref_pxls = pos_cam_in_ref_pxls.astype(int)

        pos_cam_in_ref_m = pos_cam_in_ref_pxls / self.pxl_to_m_ref

        self.current_pos = pos_cam_in_ref_m
        self.current_angle = angle

        self.get_logger().info(f"angle: {angle}, pos: {pos_cam_in_ref_m}")

        self.visualization(bird_view_img, M, good, kp1)

    def draw_2D_axis(self, image, pos_pxls, angle):
        img = image.copy()
        cv2.circle(img, tuple(pos_pxls), 10, (255,0,0), -1)

        # axis
        axis_len = 100
        x_axis = np.array([axis_len, 0])
        y_axis = np.array([0, axis_len])
        x_axis_rot = np.matmul(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), x_axis)
        y_axis_rot = np.matmul(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), y_axis)
        x_axis_rot = x_axis_rot.astype(int)
        y_axis_rot = y_axis_rot.astype(int)
        x_axis_rot = x_axis_rot + pos_pxls
        y_axis_rot = y_axis_rot + pos_pxls
        cv2.line(img, tuple(pos_pxls), tuple(x_axis_rot), (0,255,0), 2)
        cv2.line(img, tuple(pos_pxls), tuple(y_axis_rot), (0,0,255), 2)
        return img
    

    def draw_fps(self, image):
            # compute fps
        fps = 1 / (time.time() - self.time_last_image)
        self.time_last_image = time.time()

        # draw fps
        cv2.putText(image, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    

    def load_ref_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    

    def get_affine_transform_bird_to_ref(self, bird_view):

        MIN_MATCH_COUNT = 10

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift_detector.detectAndCompute(bird_view, None)

        if self.enable_masking:
            mask = self.make_mask()
            # ref_img_copy = cv2.bitwise_and(self.ref_img, mask)

            self.ref_kp, self.ref_des =  self.sift_detector.detectAndCompute(self.ref_img, mask)

            # cv2.imshow("mask", ref_img_copy)
            # cv2.waitKey(1)

        
        matches = self.flann_matcher.knnMatch(self.ref_des, des1, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        M=None
        ok = False
        if len(good)>MIN_MATCH_COUNT:
            # self.get_logger().info("Enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            src_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.ref_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            matchesMask = mask.ravel().tolist()
            ok = True
            if M is not None:
                M = np.concatenate([M, np.array([[0,0,1]])], axis=0) # TODO remove
        else:
            self.get_logger().info("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
        
        if self.enable_match_viz and ok:
            matches_img = self.draw_matches(bird_view, kp1, self.ref_kp, matchesMask, good)
            matches_img = cv2.resize(matches_img, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("Matches", matches_img)
            cv2.waitKey(1)
        
        return ok, M, matchesMask, kp1, self.ref_kp, good


    def draw_matches(self, bird_view, kp1, kp2, matchesMask, good):

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        
        img_ret = cv2.drawMatches(self.ref_img, kp2, bird_view, kp1,good, None, **draw_params)

        return img_ret
    
    def get_angle(self, M):
        # get angle from homography matrix
        angle = np.arctan2(M[1,0], M[0,0])
        return angle
    
    
    def visualization(self, bird_view=None, M=None, good=None, kp2=None):

        if bird_view is None or M is None or good is None or kp2 is None:
            if self.img_viz is None:
                return

            # draw 'not enough matches' on added_image
            self.get_logger().info("VIZ: Not enough matches")
            cv2.putText(self.img_viz, "Not enough matches", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Viz", self.img_viz)
            cv2.waitKey(1)
            return

        bird_view_in_ref = cv2.warpPerspective(bird_view, M, self.ref_img.shape[::-1])
        self.img_viz = cv2.addWeighted(bird_view_in_ref,0.5,self.ref_img,0.5,0)

        self.img_viz = cv2.cvtColor(self.img_viz, cv2.COLOR_GRAY2BGR)

        # good_matches_poses = [] # TODO il faudrait les transformer
        # for m in good:
        #     good_matches_poses.append(kp2[m.trainIdx].pt)

        # good_matches_poses = np.array(good_matches_poses)

        # # draw matches on added_image
        # for pose in good_matches_poses:
        #     cv2.circle(self.img_viz, tuple(pose.astype(int)), 3, (0,255, 0), -1)

        if self.pts_mask_viz is not None:
            self.pts_mask_viz = self.pts_mask_viz.astype(int)
            cv2.polylines(self.img_viz, [self.pts_mask_viz], True, (0,0,255), 2)
        
        # draw origin cam
        pos_cam_in_ref_pxls = np.matmul(M, self.pos_cam_in_bird_view_pxls)
        pos_cam_in_ref_pxls = pos_cam_in_ref_pxls[:2] / pos_cam_in_ref_pxls[2]
        pos_cam_in_ref_pxls = pos_cam_in_ref_pxls.astype(int)
        angle = self.get_angle(M)
        angle += np.pi # TODO enlever
        self.img_viz = self.draw_2D_axis(self.img_viz, pos_cam_in_ref_pxls, angle)

        # draw origin ref
        self.img_viz = self.draw_2D_axis(self.img_viz, np.array([0,0]), 0)
        
        self.draw_fps(self.img_viz)

        cv2.imshow("Viz", self.img_viz)
        cv2.waitKey(1)




def main(args=None):
    rclpy.init(args=args)

    visual_localization = VisualLocalizationNode()

    rclpy.spin(visual_localization)

    visual_localization.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()