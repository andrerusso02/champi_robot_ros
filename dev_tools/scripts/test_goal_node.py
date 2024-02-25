#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from math import cos, sin



class TestGoal(Node):

    def __init__(self):

        super().__init__('test_goal_node')
    



        # pub pose stamped
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.goals = [[1.5, 1., 0.], [2.0, 1., 3.14]]
        self.i_goal = 1

        timer_period = 6  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.timer_callback()



    def timer_callback(self):

        goal = PoseStamped()
        goal.pose.position.x = self.goals[self.i_goal][0]
        goal.pose.position.y = self.goals[self.i_goal][1]
        goal.pose.position.z = 0.
        goal.pose.orientation.x = 0.
        goal.pose.orientation.y = 0.
        goal.pose.orientation.z = sin(self.goals[self.i_goal][2]/2)
        goal.pose.orientation.w = cos(self.goals[self.i_goal][2]/2)

        self.pub.publish(goal)
        self.i_goal += 1
        if self.i_goal == len(self.goals):
            self.i_goal = 0
        print("Published goal")


        

def main(args=None):
    rclpy.init(args=args)
    test_goal_node = TestGoal()
    rclpy.spin(test_goal_node)
    test_goal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()