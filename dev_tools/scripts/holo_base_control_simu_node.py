#!/usr/bin/env python3

from math import sqrt, cos, sin
import time

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped


FPS = 50


class HoloBaseControlDummy(Node):

    def __init__(self):
        super().__init__('holo_base_control_dummy_node')

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.pub = self.create_publisher(Odometry, '/odom', 10)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        timer_period = 1/FPS  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Node variables
        self.latest_cmd_vel = [0., 0., 0.]
        self.current_vel = [0., 0., 0.]

        self.current_pose = [0., 0., 0.]

        self.first_time = True

        self.speed_wheel0 = 0
        self.speed_wheel1 = 0
        self.speed_wheel2 = 0

        self.robot_radius = 0.1  # TODO


    def listener_callback(self, msg):
        self.latest_cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]

    def timer_callback(self):
        self.cmd_vel_to_wheels()
        self.wheels_to_current_vel()
        self.update_pose()

        # Publish the odometry
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.twist.twist.linear.x = self.current_vel[0]
        odom.twist.twist.linear.y = self.current_vel[1]
        odom.twist.twist.angular.z = self.current_vel[2]
        odom.pose.pose.position.x = self.current_pose[0]
        odom.pose.pose.position.y = self.current_pose[1]
        odom.pose.pose.position.z = 0.
        odom.pose.pose.orientation.x = 0.
        odom.pose.pose.orientation.y = 0.
        odom.pose.pose.orientation.z = sin(self.current_pose[2] / 2)
        odom.pose.pose.orientation.w = cos(self.current_pose[2] / 2)
        self.pub.publish(odom)

        # Broadcast the transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.current_pose[0]
        t.transform.translation.y = self.current_pose[1]
        t.transform.translation.z = 0.
        t.transform.rotation.x = 0.
        t.transform.rotation.y = 0.
        t.transform.rotation.z = sin(self.current_pose[2] / 2)
        t.transform.rotation.w = cos(self.current_pose[2] / 2)
        self.tf_broadcaster.sendTransform(t)


    def wheels_to_current_vel(self):
        # Convert from WHEEL speeds --> LINEAR and ANGULAR speeds
        self.current_vel[0] = -(self.speed_wheel0 + self.speed_wheel1 - 2*self.speed_wheel2)
        self.current_vel[1] = 1/3*(-self.speed_wheel0*sqrt(3) + self.speed_wheel1*sqrt(3))
        self.current_vel[2] = (1 / (self.robot_radius)) * \
            (-self.speed_wheel0 - self.speed_wheel1 + self.speed_wheel2)

    def cmd_vel_to_wheels(self):
        # Convert from LINEAR and ANGULAR speeds --> WHEEL speeds
        cmd_vitesse_roue0 = 0.5 * \
            self.latest_cmd_vel[0] - sqrt(3) / 2 * self.latest_cmd_vel[1] - \
            self.robot_radius * self.latest_cmd_vel[2]
        cmd_vitesse_roue1 = 0.5 * \
            self.latest_cmd_vel[0] + sqrt(3) / 2 * self.latest_cmd_vel[1] - \
            self.robot_radius * self.latest_cmd_vel[2]
        cmd_vitesse_roue2 = self.latest_cmd_vel[0] - \
            self.robot_radius * self.latest_cmd_vel[2]


        # limit the speeds
        accel_roue_0 = cmd_vitesse_roue0 - self.speed_wheel0
        accel_roue_1 = cmd_vitesse_roue1 - self.speed_wheel1
        accel_roue_2 = cmd_vitesse_roue2 - self.speed_wheel2
        
        abs_accel_roue_0 = abs(accel_roue_0)
        abs_accel_roue_1 = abs(accel_roue_1)
        abs_accel_roue_2 = abs(accel_roue_2)
        abs_accel_roues = [abs_accel_roue_0, abs_accel_roue_1, abs_accel_roue_2]

        MAX_ACCEL_PER_CYCLE = 0.05 # TODO

        if abs_accel_roue_0 < MAX_ACCEL_PER_CYCLE and abs_accel_roue_1 < MAX_ACCEL_PER_CYCLE and abs_accel_roue_2 < MAX_ACCEL_PER_CYCLE:
            # acceleration requested is ok, no need to accelerate gradually.
            self.speed_wheel0 = cmd_vitesse_roue0
            self.speed_wheel1 = cmd_vitesse_roue1
            self.speed_wheel2 = cmd_vitesse_roue2
        else:
            speed_ratio = MAX_ACCEL_PER_CYCLE / max(abs_accel_roues)
            self.speed_wheel0 += speed_ratio * accel_roue_0
            self.speed_wheel1 += speed_ratio * accel_roue_1
            self.speed_wheel2 += speed_ratio * accel_roue_2


    def update_pose(self):
        # Update current_pose_ by integrating current_vel_

        if self.first_time:
            self.last_time = time.time()
            self.first_time = False
            return

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Robot speed is expressed relative to the robot frame, so we need to transform it to the world frame
        self.current_pose[0] += self.current_vel[0] * cos(self.current_pose[2]) * dt - self.current_vel[1] * sin(self.current_pose[2]) * dt
        self.current_pose[1] += self.current_vel[0] * sin(self.current_pose[2]) * dt + self.current_vel[1] * cos(self.current_pose[2]) * dt
        self.current_pose[2] += self.current_vel[2] * dt


def main(args=None):
    rclpy.init(args=args)

    holo_teleop_joy = HoloBaseControlDummy()

    rclpy.spin(holo_teleop_joy)

    holo_teleop_joy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()