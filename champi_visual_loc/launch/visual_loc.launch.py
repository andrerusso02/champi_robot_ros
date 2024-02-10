from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():

    # ros2 run tf2_ros static_transform_publisher 0 0 0.395 -1.509 3.14159 -1 base_link camera

    return LaunchDescription([

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0.395', '-1.509', '3.14159', '-1', 'base_link', 'camera']
        ),
        Node(
            package='champi_visual_loc',
            executable='visual_loc_node',
            name='visual_loc_node',
            output='screen'
        ),
    ])