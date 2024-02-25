#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import Twist
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from champi_navigation.kinematic_models import Robot_Kinematic_Model, Obstacle_static_model, Table_static_model
import champi_navigation.avoidance as avoidance
import champi_navigation.gui as gui

from icecream import ic
from dijkstar import Graph
from math import pi, atan2, cos, sin
from shapely import Point


WIDTH, HEIGHT = 900, 600  # window
TABLE_WIDTH, TABLE_HEIGHT = 3, 2  # Table size in m
FPS = 50
OFFSET = 0.15 # TODO rayon du self.robot, à voir Etienne

class PoseControl(Node):

    def __init__(self):
        super().__init__('pose_control_node')


        self.robot = Robot_Kinematic_Model(TABLE_WIDTH=TABLE_WIDTH, TABLE_HEIGHT=TABLE_HEIGHT,FPS=FPS)
        self.obstacle = Obstacle_static_model(center_x=1, center_y= 1, width= 0.1, height= 0.1,offset=OFFSET)
        self.table = Table_static_model(TABLE_WIDTH, TABLE_HEIGHT, offset=OFFSET)

        self.viz = True
        self.gui = gui.Gui(self.robot, self.obstacle, self.table)


        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.markers_pub_obstacle = self.create_publisher(Marker, "/visualization_marker_obstacle", 10)
        self.markers_pub_obstacle_offset = self.create_publisher(Marker, "/visualization_marker_obstacle_offset", 10)
        self.markers_pub_path = self.create_publisher(Marker, "/visualization_marker_path", 10)

        # Subscribe to goal pose
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.goal_sub  # prevent unused variable warning # TODO
        self.latest_goal = None


        timer_period = 1/FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def goal_callback(self, msg):
        self.latest_goal = msg

    def timer_callback(self):
        self.update()

    def update_robot_pose_from_tf(self):
        t = None
        try:
            t = self.tf_buffer.lookup_transform(
                "odom",
                "base_link",
                rclpy.time.Time())
            
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {"odom"} to {"base_link"}: {ex}')
            return

        self.robot.pos[0] = t.transform.translation.x
        self.robot.pos[1] = t.transform.translation.y
        
        q = t.transform.rotation
        self.robot.pos[2] = 2*atan2(q.z, q.w)

        # Conversion entre -pi et pi
        if self.robot.pos[2] > pi:
            self.robot.pos[2] -= 2*pi
        if self.robot.pos[2] < -pi:
            self.robot.pos[2] += 2*pi

        self.robot.robot_positions.append(self.robot.pos.copy())

    def check_goal_reached(self):
        error_max_lin = 1
        error_max_ang = 0.01

        if (len(self.robot.goals_positions))==0:
            return True

        self.robot.current_goal = self.robot.goals_positions[0]

        if abs(self.robot.pos[0] - self.robot.current_goal[0]) < error_max_lin and abs(self.robot.pos[1] - self.robot.current_goal[1]) < error_max_lin:
            if self.check_angle(self.robot.pos[2], self.robot.current_goal[2], error_max_ang):
                self.goto_next_goal()
                ic("GOAL REACHED")

                self.robot.goal_reached = True

    def goto(self, x, y, theta):
        # if it's shorter to turn in the other direction, we do it
        # TODO not always working
        error_theta = theta - self.robot.pos[2]
        if error_theta > pi:
            error_theta -= 2*pi
        elif error_theta < -pi:
            error_theta += 2*pi
        self.robot.angular_speed = self.robot.pid_pos_theta.update(error_theta)

        # # PID
        self.robot.linear_speed = [
            self.robot.pid_pos_x.update(x - self.robot.pos[0]),
            self.robot.pid_pos_y.update(y - self.robot.pos[1])
        ]


    def goto_next_goal(self):
        # called when a goal is reached to set the new current goal
        self.robot.goal_reached = False
        self.robot.has_finished_rotation = False
        if len(self.robot.goals_positions)>0:
            self.robot.goals_positions.pop(0)
        if len(self.robot.goals_positions)>0:
            self.robot.current_goal = self.robot.goals_positions[0]
        

    def check_angle(self, angle1, angle2, error_max):
        # check that the angle error is less than error_max
        error = abs(angle1 - angle2)
        if (abs(2*pi-error) < 0.01):
            error = 0
        return error < error_max


    def recompute_path(self, obstacle, table, goal_pos=None):
        if goal_pos is None:
            if len(self.robot.goals_positions) > 0:
                theta = self.robot.goals_positions[-1][2] # use the theta of the goal for each point
                goal = Point(self.robot.goals_positions[-1][0],self.robot.goals_positions[-1][1])
            else:
                return
        else:
            theta = goal_pos[2]
            goal = Point(goal_pos[0],goal_pos[1])

        start = Point(self.robot.pos[0],self.robot.pos[1])
        self.robot.graph, self.robot.dico_all_points = avoidance.create_graph(start, goal, obstacle.expanded_obstacle_poly, table.expanded_poly)
        path = avoidance.find_avoidance_path(self.robot.graph, 0, 1)
        if path is not None:
            self.robot.path_nodes = path.nodes # mais en soit renvoie aussi le coût
            goals = []

            for p in self.robot.path_nodes[1:]: # we don't add the start point
                goals.append([float(self.robot.dico_all_points[p][0]),float(self.robot.dico_all_points[p][1]), theta])
            self.robot.goals_positions = goals

    def rviz_draw(self):

        # OBSTACLE marker
        marker_obstacle = Marker()
        marker_obstacle.header.frame_id = "base_link"
        marker_obstacle.type = Marker.CUBE
        marker_obstacle.action = Marker.ADD

        marker_obstacle.scale.x = float(self.obstacle.width)
        marker_obstacle.scale.y = float(self.obstacle.height)
        marker_obstacle.scale.z = 0.3

        marker_obstacle.color.a = 1.0
        marker_obstacle.color.r = 0.5
        marker_obstacle.color.g = 0.0
        marker_obstacle.color.b = 0.0
        marker_obstacle.pose.position.x = float(self.obstacle.center_x)
        marker_obstacle.pose.position.y = float(self.obstacle.center_y)
        marker_obstacle.pose.position.z = 0.0

        self.markers_pub_obstacle.publish(marker_obstacle)

        # OFFSET OBSTACLE MARKER
        marker_obstacle_offset = Marker()
        marker_obstacle_offset.header.frame_id = "base_link"
        marker_obstacle_offset.type = Marker.CUBE
        marker_obstacle_offset.action = Marker.ADD

        marker_obstacle_offset.scale.x = float(self.obstacle.width+2*OFFSET)
        marker_obstacle_offset.scale.y = float(self.obstacle.height+2*OFFSET)
        marker_obstacle_offset.scale.z = 0.05

        marker_obstacle_offset.color.a = 1.0
        marker_obstacle_offset.color.r = 1.0
        marker_obstacle_offset.color.g = 1.0
        marker_obstacle_offset.color.b = 0.0
        marker_obstacle_offset.pose.position.x = float(self.obstacle.center_x)
        marker_obstacle_offset.pose.position.y = float(self.obstacle.center_y)
        marker_obstacle_offset.pose.position.z = 0.0

        self.markers_pub_obstacle_offset.publish(marker_obstacle_offset)

        # path markers
        if self.robot.goals_positions is not None:
            marker_path = Marker(type=Marker.LINE_STRIP, ns='points_and_lines', action=Marker.ADD)
            marker_path.header.frame_id = "base_link"
            marker_path.scale.x = 0.05
            marker_path.color.a = 1.0
            marker_path.color.r = 0.0
            marker_path.color.g = 1.0
            marker_path.color.b = 0.0
            marker_path.points = []

            # add current pos if odd
            for i in range(len(self.robot.goals_positions)-1):
                p1 = self.robot.goals_positions[i]
                p2 = self.robot.goals_positions[i+1]
                marker_path.points.append(geometry_msgs.msg.Point(x=float(p1[0]), y=float(p1[1]), z=0.0))
                marker_path.points.append(geometry_msgs.msg.Point(x=float(p2[0]), y=float(p2[1]), z=0.0))
                # si pas le dernier on ajoute encore
                if i != len(self.robot.goals_positions)-2:
                    marker_path.points.append(geometry_msgs.msg.Point(x=float(p2[0]), y=float(p2[1]), z=0.0))

            # publish the markers
            self.markers_pub_path.publish(marker_path)

    def update(self):

        if self.latest_goal is not None:
            x = self.latest_goal.pose.position.x
            y = self.latest_goal.pose.position.y
            q = self.latest_goal.pose.orientation
            theta = 2*atan2(q.z, q.w)
            self.robot.goals_positions.append([x, y, theta])
            self.latest_goal = None

        self.update_robot_pose_from_tf()

        self.check_goal_reached()

        self.goto(self.robot.current_goal[0],
        self.robot.current_goal[1],
        self.robot.current_goal[2])
                
        self.recompute_path(self.obstacle, self.table)
    
        # publish the velocity (expressed in the base_link frame)
        twist = Twist()
        twist.linear.x = self.robot.linear_speed[0] * cos(self.robot.pos[2]) + self.robot.linear_speed[1] * sin(self.robot.pos[2])
        twist.linear.y = -self.robot.linear_speed[0] * sin(self.robot.pos[2]) + self.robot.linear_speed[1] * cos(self.robot.pos[2])
        twist.angular.z = self.robot.angular_speed
        self.cmd_vel_pub.publish(twist)

        if self.viz:
            self.rviz_draw()
            self.gui.update()


        """ENVOI MESSAGE ROS CMD_VEL"""
        """LIRE POSE GOAL, POLY ROBOT ADVERSE, ODOM"""


def main(args=None):
    rclpy.init(args=args)
    pose_control_node = PoseControl()
    rclpy.spin(pose_control_node)
    pose_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()