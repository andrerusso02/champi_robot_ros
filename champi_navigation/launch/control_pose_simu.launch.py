import os
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():


    base_control_simu_node = Node(
            package='dev_tools',
            executable='holo_base_control_simu_node.py',
            name='holo_base_control_simu_node',
            output='screen',
    )

    pose_control_node = Node(
            package='champi_navigation',
            executable='nav_node.py',
            name='nav_node',
            output='screen',
    )

    return LaunchDescription([
        base_control_simu_node,
        pose_control_node,
    ])