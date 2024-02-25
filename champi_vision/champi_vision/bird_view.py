import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from icecream import ic

class BirdView:

    def __init__(self, K, T_cam_to_work_plane, pos_start_work_plane, pos_end_work_plane, resolution=20):
        """
        Initialize the BirdView object.

        Parameters:
        K (numpy.ndarray): The camera calibration matrix (3x3).
        T_cam_to_work_plane (numpy.ndarray): The transformation matrix from camera to work plane (4x4).
        pos_start_work_plane (tuple): The start position in the work plane (x, y) in meter.
        pos_end_work_plane (tuple): The end position in the work plane (x, y) in meter. Must be greater than pos_start_work_plane.
        """
        self._K = K
        self._T_cam_to_work_plane = T_cam_to_work_plane
        self._resolution = resolution # 1m = <resolution> px

        self._pos_start_work_plane = np.array(pos_start_work_plane)
        self._pos_end_work_plane = np.array(pos_end_work_plane)

        self._img_size_px = (self._pos_end_work_plane - self._pos_start_work_plane) * self._resolution

        ic(self._img_size_px)
        self._img_size_px = self._img_size_px.astype(np.int32)

        self._H = None
        self._init_projection_parameters()

    
    def _init_projection_parameters(self):

        pts_work_plane = np.array([[0., 0., 0., 1.],
                       [1., 0., 0., 1.],
                       [0., 1., 0., 1.],
                       [1., 1., 0., 1.]])
        pts_work_plane = pts_work_plane.T

        pts_cam = np.linalg.inv(self._T_cam_to_work_plane) @ pts_work_plane

        pts_sensor = pts_cam[:2] / pts_cam[2]
        pts_sensor = np.vstack((pts_sensor, np.ones((1,4))))

        pts_img = self._K @ pts_cam[:3]
        pts_img = pts_img / pts_img[2]

        M_workplane_real_to_img = self.compute_M_workplane_real_to_img(self._pos_end_work_plane, self._resolution)

        pts_work_plane_h = np.vstack((pts_work_plane[:2], np.ones((1,4))))
        pts_work_plane_px = M_workplane_real_to_img @ pts_work_plane_h
        pts_work_plane_px = pts_work_plane_px / pts_work_plane_px[2]

        ic(pts_img[:2].T.astype(np.float32))
        ic(pts_work_plane_px[:2].T.astype(np.float32))

        self._H = cv2.getPerspectiveTransform(pts_img[:2].T.astype(np.float32), pts_work_plane_px[:2].T.astype(np.float32))

    
    def compute_M_workplane_real_to_img(self, end_point, resolution):
        # start_point: [x, y]
        # end_point: [x, y]
        # resolution: [x, y]
        # return: M matrix

        # translation from origin to end_point
        T = np.array([[1, 0, -end_point[0]],
                    [0, 1, -end_point[1]],
                    [0, 0, 1]])
        # scaling
        S = np.array([[resolution, 0, 0],
                    [0, resolution, 0],
                    [0, 0, 1]])
        # rotation 90 degrees
        R = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
        # reverse y
        R_y = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
        
        # Compute M
        M = S @ R_y @ R @ T

        return M
    
    def project_img_to_bird(self, img):
        """
        Project the source image to bird's view.

        Parameters:
        img (numpy.ndarray): The source image.

        Returns:
        numpy.ndarray: The bird's view image.
        """
        return cv2.warpPerspective(img, self._H, self._img_size_px)
    
    def project_work_plane_pt_to_source_img(self, pt):
        """
        Convert a point in the work plane to a point in the source image.

        Parameters:
        pt (numpy.ndarray): The point in the work plane.

        Returns:
        numpy.ndarray: The point in the source image.
        """
        pt = pt - self._pos_start_work_plane
        pt = pt * self._resolution
        pt = np.hstack((pt, 1.))
        pt = np.linalg.inv(self._H) @ pt
        if pt[2] == 0:
            return None
        pt = pt / pt[2]
        return pt[:2]
    
    def project_source_img_pt_to_work_plane(self, pt):
        """
        Convert a point in the source image to a point in the work plane.

        Parameters:
        pt (numpy.ndarray): The point in the source image (x, y) in pixel.

        Returns:
        numpy.ndarray: The point in the work plane (x, y) in meter.
        """
        pt = np.hstack((pt, 1.))
        pt = self._H @ pt
        if pt[2] == 0:
            return None
        pt = pt / pt[2]
        pt = pt[:2]
        pt = pt / self._resolution
        pt = pt + self._pos_start_work_plane
        return pt
    
    def get_work_plane_pt_in_bird_img(self, pt):
        """
        Convert a point in the work plane to a point in the source image.

        Parameters:
        pt (numpy.ndarray): The point in the work plane (x, y) in meter.

        Returns:
        numpy.ndarray: The point in the source image (x, y) in pixel.
        """
        pt = pt - self._pos_start_work_plane
        pt = pt * self._resolution
        return pt
    
    def get_bird_img_pt_in_work_plane(self, pt):
        """
        Convert a point in the bird's view image to a point in the work plane.

        Parameters:
        pt (numpy.ndarray): The point in the bird's view image (x, y) in pixel.

        Returns:
        numpy.ndarray: The point in the work plane (x, y) in meter.
        """
        pt = pt / self._resolution
        pt = pt + self._pos_start_work_plane
        return pt


if __name__ == '__main__':
    # test
    img = cv2.imread('image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img, cmap='gray')
    plt.show()

    K = np.array([  [1124.66943359375, 0.0, 505.781982421875],
                [0.0, 1124.6165771484375, 387.8110046386719],
                [0.0, 0.0, 1.0]])

    T_cam_to_work_plane = np.array([[ 0.99991969 , 0.01149682 ,-0.00533206 , 0.05419943],
        [ 0.00510958 , 0.01929436 , 0.99980079 , 1.96159697],
        [ 0.0115974  ,-0.99974774  ,0.01923406  ,1.55057154],
        [ 0.          ,0.          ,0.          ,1.        ]])

    bv = BirdView(K, T_cam_to_work_plane, (-20, 5), (20, 50))
    
    time_start = time.time()

    img_bird = bv.project_img_to_bird(img)

    time_end = time.time()
    print("time: ", time_end - time_start)

    pt_px_cam = bv.project_work_plane_pt_to_source_img(np.array([20., 50.]))
    print(pt_px_cam)

    pt_px_bird = bv.get_work_plane_pt_in_bird_img(np.array([-20., 5.]))
    print(pt_px_bird)

    pt_work_plane = bv.get_bird_img_pt_in_work_plane(np.array([800., 900.]))
    print(pt_work_plane)

    pt_work_plane = bv.project_source_img_pt_to_work_plane(np.array([979.47613383, 451.36251473]))
    print(pt_work_plane)

    plt.imshow(img_bird, cmap='gray')
    plt.show()