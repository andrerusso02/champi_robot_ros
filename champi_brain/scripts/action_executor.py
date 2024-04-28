#!/usr/bin/env python3

from robot_navigator import Robot_Navigator

import time
from typing import Tuple
from enum import Enum

from std_msgs.msg import String, ColorRGBA

from rclpy.node import Node

import plants_taker_api
from utils import draw_rviz_markers
from utils import State, StateTakingPlants



class Action_Executor():
    def __init__(self,strategy_node: Node) -> None:
        # check that everything is ready
        self.wait_for_ready()
        self.robot_navigator = Robot_Navigator()
        self.strategy_node = strategy_node

        # PLANTS STATE MACHINE
        self.state_taking_plants = None

        # MAIN STATE MACHINE
        self.state = State.NO_ACTION
        self.current_action = None

    def new_action(self, action:dict):
        if action["type"] == "prendre_plantes":
            self.state = State.PLANTES
            self.state_taking_plants = StateTakingPlants.COMPUTE_FRONT_POSE

    def update(self, current_action: dict):
        # print("update executor")
        self.current_action = current_action

        if self.state == State.PLANTES:
            self.update_taking_plants()
        # elif self.state == State.POSE_PLANTES:
        #     self.update_pose_plantes()
        # elif self.state == State.PANNEAU:
        #     self.update_panneau()
        # elif self.state == State.RETOUR:
        #     self.update_retour()

    def update_taking_plants(self):
        if self.state_taking_plants == StateTakingPlants.COMPUTE_FRONT_POSE:
            print("\tComputing front pose from plants")
            # compute the front pose from the plants
            self.front_pose_from_plants = plants_taker_api.compute_front_pose_from_plants_position(self.current_action["pose"], self.strategy_node.robot_pose)
            self.state_taking_plants = StateTakingPlants.NAVIGATE_TO_FRONT_POSE
        elif self.state_taking_plants == StateTakingPlants.NAVIGATE_TO_FRONT_POSE:
            print("\tNavigating to front pose from plants: ", self.front_pose_from_plants)
            # navigate to the front pose
            self.robot_navigator.navigate_to(self.front_pose_from_plants, self.current_action["time"])
            self.strategy_node.robot_pose = self.front_pose_from_plants
            self.state_taking_plants = StateTakingPlants.DETECT_PLANTS
        elif self.state_taking_plants == StateTakingPlants.DETECT_PLANTS:
            print("\tDetecting plants")
            # wait for the plants to be detected
            self.plants_positions = plants_taker_api.wait_and_detect_plants(self.current_action["pose"], self.strategy_node.robot_pose)
            draw_rviz_markers(self.strategy_node, self.plants_positions, "/markers_plants_detected",ColorRGBA(r=0., g=1., b=0., a=1.), 0.06)
            draw_rviz_markers(self.strategy_node, [self.front_pose_from_plants], "/markers_selected_action",ColorRGBA(r=1., g=0., b=0., a=1.),0.05)
            self.state_taking_plants = StateTakingPlants.COMPUTE_TRAJECTORY
        elif self.state_taking_plants == StateTakingPlants.COMPUTE_TRAJECTORY:
            print("\tComputing trajectory")
            # compute the trajectory points
            self.front_pose_from_plants = (self.front_pose_from_plants[0], self.front_pose_from_plants[1], self.front_pose_from_plants[2])
            self.trajectory_points = plants_taker_api.compute_trajectory_points(self.plants_positions, self.front_pose_from_plants, self.current_action["pose"])
            self.state_taking_plants = StateTakingPlants.NAVIGATE_TO_TRAJECTORY_1
        elif self.state_taking_plants == StateTakingPlants.NAVIGATE_TO_TRAJECTORY_1:
            print("\tNavigating to first point of trajectory")
            # navigate to the first point of the trajectory
            self.robot_navigator.navigate_to(self.trajectory_points[0], self.current_action["time"])
            self.publish_on_CAN("be_ready_to_grab_plants")
            self.state_taking_plants = StateTakingPlants.NAVIGATE_TO_TRAJECTORY_2
        elif self.state_taking_plants == StateTakingPlants.NAVIGATE_TO_TRAJECTORY_2:
            print("\tNavigating to second point of trajectory")
            # navigate to the second point of the trajectory
            self.robot_navigator.navigate_to(self.trajectory_points[1], self.current_action["time"])
            self.publish_on_CAN("finished_grabbing_plants")
            self.state_taking_plants = None
            self.state = State.NO_ACTION
            self.front_pose_from_plants = None
            self.plants_positions = None
            self.trajectory_points = None
            self.state = State.ACTION_FINISHED


    def publish_on_CAN(self, message: str):
        # publish on the topic "/CAN" to be forwarded by CAN by the API
        msg = String()
        msg.data = message
        self.strategy_node.CAN_publisher.publish(msg)   

    def wait_for_ready(self):
        # check on CAN/topics that every of the 3 stm boards are ready
        # TODO
        return True

    def stop_effectors(self):
        self.publish_on_CAN("stop_effectors")