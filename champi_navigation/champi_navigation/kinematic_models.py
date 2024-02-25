from champi_navigation.pid import PID
import champi_navigation.math_bind as math_bind 

from typing import List
from math import pi, sqrt
from icecream import ic
from shapely import Point, Polygon


class Robot_Kinematic_Model():
    def __init__(self, TABLE_WIDTH, TABLE_HEIGHT, FPS) -> None:
        self.pos = [TABLE_WIDTH/2,  # x, m
                    TABLE_HEIGHT/2,  # y, m
                    0]  # theta, rad between -pi and pi
        self.wheel_radius = 5.8
        self.wheel_width = 5
        self.robot_radius = 15
        self.robot_positions = []

        self.linear_speed = [0, 0]  # m/s
        self.angular_speed = 0  # rad/s
        self.has_finished_rotation = False

        self.speed_wheel0 = 0
        self.speed_wheel1 = 0
        self.speed_wheel2 = 0

        self.MAX_ACCEL_PER_CYCLE = 500/FPS # rotation/s/cycle

        self.current_goal = None
        self.goal_reached = True
        self.goals_positions = [[TABLE_WIDTH/2, TABLE_HEIGHT/2, 0],
                                ]  # [x, y, theta]
        
        self.graph = None
        self.dico_all_points = {}
        self.path_nodes = None

        self.max_ang_speed = 1.2  # pi/2  # rad/s

        # just a P
        P_lin = 0.2
        self.pid_pos_x = PID(P_lin, 0, 0, 1/FPS)
        self.pid_pos_y = PID(P_lin, 0, 0, 1/FPS)
        self.pid_pos_theta = PID(1, 0, 0, 1/FPS)
        self.delta_t = 1 / FPS  # Time between two updates

    def write_speeds(self, speeds: List) -> None:
        self.speed_wheel0 = speeds[0]
        self.speed_wheel1 = speeds[1]
        self.speed_wheel2 = speeds[2]
    # TODO A CHANGER POUR LINEAR & ANGULAR 

class Obstacle_static_model():
    def __init__(self, center_x, center_y, width, height, offset) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.polygon = Polygon([Point(center_x+width/2,center_y+height/2),
                               Point(center_x+width/2,center_y-height/2),
                               Point(center_x-width/2,center_y-height/2),
                               Point(center_x-width/2,center_y+height/2)])
        self.expanded_obstacle_poly = math_bind.expand(self.polygon, offset)

class Table_static_model():
    def __init__(self, width, height, offset) -> None:
        self.width = width
        self.height = height
        self.polygon = Polygon([Point(0,0),
                                Point(width,0),
                                Point(width, height),
                                Point(0, height)])
        self.expanded_poly = math_bind.expand(self.polygon, -offset)