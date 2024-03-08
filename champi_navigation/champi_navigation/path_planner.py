#!/usr/bin/env python3

from shapely import Point
from icecream import ic
from math import sin, cos

from champi_navigation import avoidance
import champi_navigation.trajectory as trajectory

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose

TABLE_WIDTH, TABLE_HEIGHT = 3, 2  # Table size in m

class PathPlanner:
    def __init__(self, world_state):

        self.trajectory_builder = trajectory.TrajectoryBuilder(world_state)
        self.world_state = world_state

        self.cmd_goal = None
        self.enable_avoidance = True

        # self.environment_state = None
        # self.robot_state = None

        # self.graph = None
        # self.dico_all_points = {}
        # self.path_nodes = None

    

    def set_cmd_goal(self, goal):
        """Set the goal of the robot (x, y, theta)
            self.update() should be called to compute a new path"""
        self.cmd_goal = goal
        ic("RECEIVED NEW CMD GOAL :")
        ic(self.cmd_goal)

    def update(self, current_time):
        """Spin once of the planning loop"""

        if self.cmd_goal is None:
            return []
        
        if self.enable_avoidance:
            cmd_path = self.compute_path_avoidance()
        else:
            cmd_path = self.compute_path_simple()
        
        ic("PATH COMPUTED :")
        ic(cmd_path)

        # convert the cmd_path [[x, y, theta],...] to a Path ros msg
        cmd_path_msg = Path()
        cmd_path_msg.header.frame_id = "odom"
        cmd_path_msg.header.stamp = current_time
        for pose in cmd_path:
            p = PoseStamped()
            p.pose.position.x = pose[0]
            p.pose.position.y = pose[1]
            p.pose.position.z = 0.
            p.pose.orientation.x = 0.
            p.pose.orientation.y = 0.
            p.pose.orientation.z = sin(pose[2]/2)
            p.pose.orientation.w = cos(pose[2]/2)
            cmd_path_msg.poses.append(p)


        # ic("PATH MSG :")
        # ic(cmd_path_msg)
        ic("\n")
        return cmd_path_msg


    def compute_path_simple(self):
        """Return a direct path from the current robot position to the goal.
        Must be called with a goal != None."""
        cmd_path = [self.robot_state.current_pose, self.cmd_goal]
        return cmd_path


    def compute_path_avoidance(self):
        """Must be called with a goal != None."""

        # ic("COMPUTE PATH AVOIDANCE")

        goal = Point(self.cmd_goal.position.x, self.cmd_goal.position.y)
        theta = self.cmd_goal.orientation.z # TODO sûr ?
        start = Point(self.world_state.get_self_robot()["pose"].position_m[0],
                      self.world_state.get_self_robot()["pose"].position_m[1])

        ic(start, goal, theta)

        self.graph, self.dico_all_points = avoidance.create_graph(start, goal, self.world_state)
        # ic("GRAPH CREATED")
        path = avoidance.find_avoidance_path(self.graph, 0, 1)
        # ic("PATH FOUND")
        
        if path is not None:
            self.path_nodes = path.nodes # note : return also the costs
            
            goals = []
            for p in self.path_nodes:
                goals.append([float(self.dico_all_points[p][0]),float(self.dico_all_points[p][1]), theta])
            return goals
        else:
            ic("No path found")
            return []