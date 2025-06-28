#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math

class InitialPositioningTest(Node):
    def __init__(self):
        super().__init__('initial_positioning_test')
        
        # Create a publisher for goals
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Create a timer to send test goals
        self.timer = self.create_timer(10.0, self.send_test_goal)
        self.goal_count = 0
        
        self.get_logger().info('Initial positioning test node started. Will send test goals every 10 seconds.')
    
    def send_test_goal(self):
        """Send a test goal to verify initial positioning."""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        # Send different goals to test different orientations
        if self.goal_count == 0:
            # Goal to the right (90 degrees)
            goal.pose.position.x = 2.0
            goal.pose.position.y = 0.0
            goal.pose.orientation.w = 1.0
            self.get_logger().info('Sending test goal 1: (2.0, 0.0) - should require ~90° rotation')
        elif self.goal_count == 1:
            # Goal to the left (-90 degrees)
            goal.pose.position.x = -2.0
            goal.pose.position.y = 0.0
            goal.pose.orientation.w = 1.0
            self.get_logger().info('Sending test goal 2: (-2.0, 0.0) - should require ~-90° rotation')
        elif self.goal_count == 2:
            # Goal behind (180 degrees)
            goal.pose.position.x = 0.0
            goal.pose.position.y = -2.0
            goal.pose.orientation.w = 1.0
            self.get_logger().info('Sending test goal 3: (0.0, -2.0) - should require ~180° rotation')
        else:
            # Goal in front (0 degrees)
            goal.pose.position.x = 0.0
            goal.pose.position.y = 2.0
            goal.pose.orientation.w = 1.0
            self.get_logger().info('Sending test goal 4: (0.0, 2.0) - should require minimal rotation')
            self.goal_count = -1  # Reset counter
        
        self.goal_pub.publish(goal)
        self.goal_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = InitialPositioningTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 