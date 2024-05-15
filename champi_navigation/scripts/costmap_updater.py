#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from icecream import ic

class ImageToCostmapNode(Node):
    def __init__(self):
        super().__init__('image_to_costmap_node')
        self.publisher_ = self.create_publisher(OccupancyGrid, 'costmap', 10)
        self.bridge = CvBridge()
        self.timer_period = 0.2  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Parameter for grid width and height (meters) and resolution (m/pixel)
        self.grid_width = self.declare_parameter('grid_width', 2.0).value
        self.grid_height = self.declare_parameter('grid_height', 3.0).value
        self.resolution = self.declare_parameter('resolution', 0.05).value

        self.robot_radius = self.declare_parameter('robot_radius', 0.2).value
        self.enemy_robot_radius = self.declare_parameter('enemy_robot_radius', 0.2).value

        # Create a black image with the specified width and height
        self.static_layer_img = np.zeros((int(self.grid_height / self.resolution), int(self.grid_width / self.resolution)), np.uint8)

        # Draw borders (robot radius)
        self.static_layer_img[0:int(self.robot_radius / self.resolution), :] = 100
        self.static_layer_img[-int(self.robot_radius / self.resolution):, :] = 100
        self.static_layer_img[:, 0:int(self.robot_radius / self.resolution)] = 100
        self.static_layer_img[:, -int(self.robot_radius / self.resolution):] = 100

        self.obstacle_layer_img = np.zeros((int(self.grid_height / self.resolution), int(self.grid_width / self.resolution)), np.uint8)


        # Subscribe to the enemy position
        self.enemy_position = None
        self.enemy_position_sub = self.create_subscription(PoseStamped, '/enemy_pose', self.enemy_position_callback, 10)


    def enemy_position_callback(self, msg):
        self.enemy_position = msg

    def timer_callback(self):

        if self.enemy_position is not None:
            self.clear_obstacle_layer()

            enemy_x = int(round(self.enemy_position.pose.position.x / self.resolution))
            enemy_y = int(round(self.enemy_position.pose.position.y / self.resolution))
            radius = int(np.ceil((self.enemy_robot_radius + self.robot_radius) / self.resolution))

            # Draw enemy robot
            cv2.circle(self.obstacle_layer_img, (enemy_x, enemy_y), radius, 100, -1)

        occupancy_img = self.combine_layers()

        occupancy_grid_msg = self.image_to_occupancy_grid(occupancy_img)
        self.publisher_.publish(occupancy_grid_msg)

    def clear_obstacle_layer(self):
        self.obstacle_layer_img = np.zeros((int(self.grid_height / self.resolution), int(self.grid_width / self.resolution)), np.uint8)

    def combine_layers(self):
        # Sum the two layers and clip the values to 100
        occupancy_img = np.clip(self.static_layer_img + self.obstacle_layer_img, 0, 100)
        return occupancy_img

    def image_to_occupancy_grid(self, img):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = 'map'
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.width = img.shape[1]
        occupancy_grid.info.height = img.shape[0]
        # ic((255 - 128 - img.flatten()).tolist())
        occupancy_grid.data = (img.flatten()).tolist()  # invert colors and flatten the image
        return occupancy_grid

def main(args=None):
    rclpy.init(args=args)

    image_to_costmap_node = ImageToCostmapNode()

    rclpy.spin(image_to_costmap_node)

    image_to_costmap_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()