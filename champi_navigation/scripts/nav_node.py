#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

from icecream import ic
from math import pi, atan2
import numpy as np

from champi_navigation.gui_v2 import GuiV2
from champi_navigation.pose_control import PoseControl
from champi_navigation.utils import Vel, RobotState
from champi_navigation.path_planner import PathPlanner


class NavigationNode(Node):

    def __init__(self):
        super().__init__('pose_control_node')

        # Parameters
        self.control_loop_period = self.declare_parameter('control_loop_period', 0.1).value
        self.enable_viz = self.declare_parameter('viz', True).value
        self.enable_avoidance = self.declare_parameter('enable_avoidance', True).value

        # Viz in Rviz
        if self.enable_viz:
            self.viz = GuiV2(self)
        else:
            self.viz = None

        self.poseControl = PoseControl(self.viz)

        self.pathPlanner = PathPlanner(self.enable_avoidance)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/cmd_path', 10)
        self.markers_pub_obstacle = self.create_publisher(Marker, "/visualization_marker_obstacle", 10)
        self.markers_pub_obstacle_offset = self.create_publisher(Marker, "/visualization_marker_obstacle_offset", 10)


        # Subscribers
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.goal_sub  # prevent unused variable warning
        
        # Timers
        self.timer = self.create_timer(self.control_loop_period, self.nav_loop_callback)

        # TF related
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Diagnostic
        self.last_time_ctrl_loop = None

        # Variables
        self.latest_goal_msg = None


    def goal_callback(self, msg):
        """Callback for the goal pose message. It is called when a new goal is received from topic."""
        self.latest_goal_msg = msg
    
    
    def get_cmd_goal(self):
        """Construct the command path for the robot""" # TODO temporary
        if self.latest_goal_msg is None:
            return None
        x = self.latest_goal_msg.pose.position.x
        y = self.latest_goal_msg.pose.position.y
        q = self.latest_goal_msg.pose.orientation
        theta = 2*atan2(q.z, q.w)
        return [x, y, theta]
        

    def nav_loop_callback(self):

        """Compute loop time"""

        # Initialize the last_time_ctrl_loop
        if self.last_time_ctrl_loop is None:
            self.last_time_ctrl_loop = self.get_clock().now()
            return
        
        # Compute the time elapsed since the last control loop
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time_ctrl_loop).nanoseconds / 1e9
        self.last_time_ctrl_loop = current_time

        """Construct the current state of the robot"""

        current_robot_pose = self.get_robot_pose_from_tf()
        if current_robot_pose is None: # TF could not be received
            return
        current_robot_speed = self.get_current_robot_speed()
        robot_state = RobotState(current_robot_pose, current_robot_speed)

        """
        Path Planning: 
            inputs:
                A. observations 
                    - robot state,
                    - environment state,
                B. Command
                    - goal (single pose, from topic)
            output:
                - cmd_path (list of poses to follow)
        """

        # Transmit the current state of the robot to path planner
        self.pathPlanner.set_robot_state(robot_state)

        # Transmit the goal to path planner
        cmd_goal = self.get_cmd_goal()
        self.pathPlanner.set_cmd_goal(cmd_goal) # Note: It can be None! Which means no goal to reach.
        
        # Call the planning loop of path planner
        cmd_path = self.pathPlanner.planning_loop_spin_once()
        
        """
        Pose Control:
            inputs:
                A. observations
                    - robot state,
                B. Command
                    - cmd_path (list of poses to follow, from path planner)
            output:
                - cmd_twist (TwistStamped)
        """

        # Transmit the current state of the robot to pose control
        self.poseControl.set_robot_state(robot_state)

        # Transmit the command path to pose control
        self.poseControl.set_cmd_path(cmd_path) # Note: It can be empty! Which means no path to follow.

        # Call the control loop of pose control
        cmd_twist = self.poseControl.control_loop_spin_once(dt)

        cmd_twist_stamped = TwistStamped()
        cmd_twist_stamped.header.stamp = current_time.to_msg()
        cmd_twist_stamped.header.frame_id = "base_link"
        cmd_twist_stamped.twist = cmd_twist

        # Publish the command
        self.cmd_vel_pub.publish(cmd_twist_stamped)

        # Publish the path. Only for visualization purposes for now.
        self.publish_path(cmd_path)
        # Publish the obstacle. Only for visualization purposes.
        self.pub_rviz_obstacles()


    def pub_rviz_obstacles(self):
        # OBSTACLE marker
        marker_obstacle = Marker()
        marker_obstacle.header.frame_id = "odom"
        marker_obstacle.type = Marker.CUBE
        marker_obstacle.action = Marker.ADD

        marker_obstacle.scale.x = float(self.pathPlanner.obstacle.width)
        marker_obstacle.scale.y = float(self.pathPlanner.obstacle.height)
        marker_obstacle.scale.z = 0.3

        marker_obstacle.color.a = 1.0
        marker_obstacle.color.r = 0.5
        marker_obstacle.color.g = 0.0
        marker_obstacle.color.b = 0.0
        marker_obstacle.pose.position.x = float(self.pathPlanner.obstacle.center_x)
        marker_obstacle.pose.position.y = float(self.pathPlanner.obstacle.center_y)
        marker_obstacle.pose.position.z = 0.0

        self.markers_pub_obstacle.publish(marker_obstacle)


        # OFFSET OBSTACLE MARKER
        marker_obstacle_offset = Marker()
        marker_obstacle_offset.header.frame_id = "odom"
        marker_obstacle_offset.type = Marker.CUBE
        marker_obstacle_offset.action = Marker.ADD

        OFFSET = self.pathPlanner.offset
        marker_obstacle_offset.scale.x = float(self.pathPlanner.obstacle.width+2*OFFSET)
        marker_obstacle_offset.scale.y = float(self.pathPlanner.obstacle.height+2*OFFSET)
        marker_obstacle_offset.scale.z = 0.05

        marker_obstacle_offset.color.a = 1.0
        marker_obstacle_offset.color.r = 1.0
        marker_obstacle_offset.color.g = 1.0
        marker_obstacle_offset.color.b = 0.0
        marker_obstacle_offset.pose.position.x = float(self.pathPlanner.obstacle.center_x)
        marker_obstacle_offset.pose.position.y = float(self.pathPlanner.obstacle.center_y)
        marker_obstacle_offset.pose.position.z = 0.0

        self.markers_pub_obstacle_offset.publish(marker_obstacle_offset)


    def get_robot_pose_from_tf(self):
        t = None
        try:
            t = self.tf_buffer.lookup_transform(
                "odom",
                "base_link",
                rclpy.time.Time())
            
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {"odom"} to {"base_link"}: {ex}')
            return None
        
        q = t.transform.rotation
        pose = np.array([t.transform.translation.x, t.transform.translation.y, 2*atan2(q.z, q.w)])

        # Conversion entre -pi et pi
        if pose[2] > pi:
            pose[2] -= 2*pi
        elif pose[2] < -pi:
            pose[2] += 2*pi
        
        return pose


    def get_current_robot_speed(self):
        return Vel(0, 0, 0) # TODO
    
    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"
        path_msg.poses = [PoseStamped() for _ in path]
        for i, pose in enumerate(path):
            path_msg.poses[i].header = path_msg.header
            path_msg.poses[i].pose.position.x = pose[0]
            path_msg.poses[i].pose.position.y = pose[1]
            path_msg.poses[i].pose.position.z = 0.
            path_msg.poses[i].pose.orientation.x = 0.
            path_msg.poses[i].pose.orientation.y = 0.
            path_msg.poses[i].pose.orientation.w = np.cos(pose[2]/2)
            path_msg.poses[i].pose.orientation.z = np.sin(pose[2]/2)
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    pose_control_node = NavigationNode()
    rclpy.spin(pose_control_node)
    pose_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()