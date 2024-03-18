import os
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    # Declare the launch options
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='True',
        description='Launch simulation'
    )

    # Get configuration file
    config_file_path = os.path.join(get_package_share_directory('champi_bringup'), 'config', 'champi.config.yaml')


    # Get the URDF file TODO faire ça dans un launch file dans champi_description plutôt
    urdf_file_path = os.path.join(get_package_share_directory('champi_description'), 'urdf', 'champi.urdf')
    urdf_content = open(urdf_file_path).read()    

    description_broadcaster = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': urdf_content}]
    )

    base_controller_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource([
            get_package_share_directory('champi_controllers'),
            '/launch/base_controller.launch.py'
        ]),
        condition=UnlessCondition(LaunchConfiguration('sim'))

    )

    # Include LDLidar with lifecycle manager launch
    ldlidar_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource([
            get_package_share_directory('ldlidar_node'),
            '/launch/ldlidar_with_mgr.launch.py'
        ]),
        launch_arguments={
            'node_name': 'ldlidar_node'
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('sim'))
    )

    lidar_simu_node = Node(
        package='dev_tools',
        executable='simu_lidar_node.py',
        name='simu_lidar_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('sim'))
    )

    base_control_simu_node = Node(
        package='dev_tools',
        executable='holo_base_control_simu_node.py',
        name='holo_base_control_simu_node',
        output='screen',
        remappings=[('/cmd_vel', '/base_controller/cmd_vel')],
        condition=IfCondition(LaunchConfiguration('sim'))
    )

    # cmd_vel multiplexer
    cmd_vel_mux_node = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen', # TODO tester output='both'
        parameters=[config_file_path],
        remappings=[('/cmd_vel_out', '/base_controller/cmd_vel')]
    )

    joy_node = Node(
        package='joy',
        executable='game_controller_node',
        name='game_controller_node',
        output='screen'
    )

    holo_teleop_joy_node = Node(
        package='dev_tools',
        executable='holo_teleop_joy_node.py',
        name='holo_teleop_joy_node',
        output='screen',
        remappings=[('/cmd_vel', '/cmd_vel_joy')]
    )

    return LaunchDescription([
        sim_arg,
        description_broadcaster,
        base_controller_launch,
        ldlidar_launch,
        lidar_simu_node,
        base_control_simu_node,
        cmd_vel_mux_node,
        joy_node,
        holo_teleop_joy_node
    ])

