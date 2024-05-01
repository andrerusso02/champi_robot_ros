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
        description='Launch simulation (true|false)',
    )

    joy_arg = DeclareLaunchArgument(
        'joy',
        default_value='False',
        description='Launch joystick (true|false)',
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

    imu_controller_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource([
            get_package_share_directory('champi_controllers'),
            '/launch/imu_controller.launch.py'
        ]),
        condition=UnlessCondition(LaunchConfiguration('sim'))
    )

  # LDROBOT LiDAR publisher node
    ldlidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='LD19',
        output='screen',
        parameters=[
            {'product_name': 'LDLiDAR_LD19'},
            {'topic_name': 'scan'},
            {'frame_id': 'base_laser'},
            {'port_name': '/dev/ttyUSB0'},
            {'port_baudrate': 230400},
            {'laser_scan_dir': True},
            {'enable_angle_crop_func': False},
            {'angle_crop_min': 135.0},
            {'angle_crop_max': 225.0}
        ],
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
        name='base_controller_simu',
        output='screen',
        parameters=[config_file_path],
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

    # Teleop
    teleop_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource([
            get_package_share_directory('champi_bringup'),
            '/launch/teleop.launch.py'
        ]),
        condition=IfCondition(LaunchConfiguration('joy'))
    )

    pub_goal_rviz_node = Node(
        package='dev_tools',
        executable='pub_goal_rviz.py',
        name='pub_goal_rviz',
        output='screen'
    )

    ukf_node = Node(
        package='robot_localization',
        executable='ukf_node',
        name='ukf',
        output='screen',
        parameters=[os.path.join(get_package_share_directory("champi_bringup"), "config", "ukf.yaml")],
        remappings=[('/cmd_vel', '/base_controller/cmd_vel_limited')]
    )

    # Static transform map -> odom
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_odom',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        sim_arg,
        joy_arg,
        description_broadcaster,
        base_controller_launch,
        imu_controller_launch,
        # ldlidar_node,
        lidar_simu_node,
        base_control_simu_node,
        cmd_vel_mux_node,
        teleop_launch,
        pub_goal_rviz_node,
        ukf_node,
        static_tf_map_odom
    ])

