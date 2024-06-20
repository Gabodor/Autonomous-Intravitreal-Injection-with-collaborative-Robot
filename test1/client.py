#!/usr/bin/env python3
import sys
from project_interfaces.srv import Pose

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage

from roboticstoolbox import quintic, mtraj

STEP_MAX = 10
FREQUENCY = 4
TIME_STEP = 1/FREQUENCY
TIME_MAX_SEC = STEP_MAX/FREQUENCY

class Ur3_controller(Node):
    def __init__(self):
        super().__init__('ur3_controller')
        self.cli = self.create_client(Pose, 'test1')
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 10)               # target_frame(Coppeliasim) / topic(Listener)
        self.subscription = self.create_subscription(
            TFMessage,
            'tf',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.check = False

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Pose.Request()

        self.trajectory_planning()
        self.move_first()
        self.move_second()

    def send_request(self, r):
        self.req.r = r
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def trajectory_planning(self):                                                          #da rivedere
        # Deve chiedere configurazione attuale end-effector e finale
        response = self.send_request(0)
        self.get_logger().info('Requesting first quaternion to compute planning')

        while not self.check:
            pass

        #q_end=(0.37158, 0.15214, 0.32538, 0.99999, 0.00039, 0.00039, 0.00039)
        q_end=(0.37158, 0.15214, 0.32538, response.x, response.y, response.z, response.w)
        self.trajectory = mtraj(quintic, self.q_start, q_end, STEP_MAX)
    
    def publish_pose(self, position, orientation):
        msg = PoseStamped()

        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation[0]                       #controlla ordine x y z w
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
    
        self.publisher.publish(msg)
        #self.get_logger().info('Publishing pose: [ position: (x:%f, y:%f, z:%f), orientation (x:%f, y:%f, z:%f, w:%f)]'
        #                        % (position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]))
        self.get_logger().info('Publishing pose')
                               
    def move_first(self):
        self.time = 0
        while self.time < STEP_MAX :
            q = self.trajectory.q[self.time]
            position = (q[0], q[1], q[2])
            orientation = (q[3], q[4], q[5], q[6])
            self.publish_pose(position, orientation)
            time.sleep(TIME_STEP)      
            self.time = self.time+1
        print("\nMove first complete\n")
    
    def move_second(self):
        while True:
            response = self.send_request(0)
            position = (0.37158, 0.15214, 0.32538)
            orientation = (response.x, response.y, response.z, response.w)

            self.get_logger().info('Requesting quaternion')
            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)        
        print("Move second complete")

    def listener_callback(self, msg):
        self.destroy_subscription(self.subscription)

        for t in msg.transforms:
            self.get_logger().info('Pose %s: [ translation: (x:%f, y:%f, z:%f), rotation: (x:%f, y:%f, z:%f, w:%f)]'
                                    % (t.child_frame_id,
                                        t.transform.translation.x,
                                        t.transform.translation.y,
                                        t.transform.translation.z,
                                        t.transform.rotation.x,
                                        t.transform.rotation.y,
                                        t.transform.rotation.z,
                                        t.transform.rotation.w))
        
        self.q_start=(-0.13088, 0.29877, 0.30323, 0.99999, 0.00039, 0.00039, 0.00039)
        self.check = True



def main():
    rclpy.init()

    ur3_controller = Ur3_controller()
    print("arrivato?")
    rclpy.spin(ur3_controller)

    ur3_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
