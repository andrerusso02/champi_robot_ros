#!/usr/bin/env python3

from math import atan2, cos, sin
import numpy as np
from geometry_msgs.msg import Twist


class Vel:
    """Velocity. It can be expressed in any frame."""
    def __init__(self, x=0., y=0., theta=0.):
        self.x = x # m/s
        self.y = y # m/s
        self.theta = theta # rad/s
    
    def init_from_mag_ang(self, mag, angle, theta):
        self.x = mag * cos(angle)
        self.y = mag * sin(angle)
        self.theta = theta

    def as_mag_ang(self):
        angle = atan2(self.y, self.x)
        mag = (self.x**2 + self.y**2)**0.5
        return np.array([mag, angle, self.theta])

    def __str__(self):
        return f'CmdVel(x={self.x}, y={self.y}, theta={self.theta})'

    def to_twist(self):
        twist = Twist()
        twist.linear.x = float(self.x)
        twist.linear.y = float(self.y)
        twist.angular.z = float(self.theta)
        return twist
    
    @staticmethod
    def to_robot_frame(robot_pose, cmd_vel):
        """Transform a velocity expressed in the base_link frame to the robot frame"""
        x = cmd_vel.x * cos(robot_pose[2]) + cmd_vel.y * sin(robot_pose[2])
        y = -cmd_vel.x * sin(robot_pose[2]) + cmd_vel.y * cos(robot_pose[2])
        theta = cmd_vel.theta
        return Vel(x, y, theta)
    
    @staticmethod
    def to_global_frame(robot_pose, cmd_vel):
        """Transform a velocity expressed in the robot frame to the base_link frame"""
        x = cmd_vel.x * cos(robot_pose[2]) - cmd_vel.y * sin(robot_pose[2])
        y = cmd_vel.x * sin(robot_pose[2]) + cmd_vel.y * cos(robot_pose[2])
        theta = cmd_vel.theta
        return Vel(x, y, theta)

class RobotState:
    def __init__(self, pose: np.array, vel: Vel):
        # TODO Etienne, est ce qu'on utilise numpy partout ou nulle part ?
        # Moi, j'aime bien numpy parce qu'il y a beaucoup de fonctions cool, 
        # c'est du cpp derrière, et c'est possible de faire du parallelisme
        # avec jit et numba moyennant quelques modifications après.
        # Mais pour juste représenter des points, y a pas un giga intérêt non plus...
        self.current_pose = pose
        self.current_vel = vel 
    

