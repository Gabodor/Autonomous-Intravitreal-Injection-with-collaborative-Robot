#!/usr/bin/env python3
import sys
from project_interfaces.srv import Pose

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from visualization_msgs.msg import  Marker

from roboticstoolbox import quintic, mtraj
import numpy as np
from transforms3d import euler

# Costant values
STEP_MAX = 10
FREQUENCY = 4
TIME_STEP = 1/FREQUENCY
TIME_MAX_SEC = STEP_MAX/FREQUENCY

class Ur3_controller(Node):
    def __init__(self):
        # Initialize class
        super().__init__('ur3_controller')
        
        # Create client(eye-tracking) to get pose
        self.cli = self.create_client(Pose, 'test1')

        # Wait for eye-tracking to start
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Pose.Request()

        # Create publisher to publish planning
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 10)
        
        # Getting initial pose
        self.get_initial_pose()

        self.trajectory_planning()
        self.move_first()
        self.move_second()

    def get_initial_pose(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        waiting_pose = True
        while waiting_pose:           
            try:
                future = self.tf_buffer.wait_for_transform_async(
                        'base_link',
                        'wrist_3_link',
                        rclpy.time.Time()
                        )
                rclpy.spin_until_future_complete(self, future)
                trans = self.tf_buffer.lookup_transform(
                    'base_link',
                    'wrist_3_link',
                    rclpy.time.Time()
                    )
                self.current_pose = (   trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
                                        trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z)
                waiting_pose = False
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
    
    def trajectory_planning(self):
        # Get eye orientation 
        w, x, y, z = self.get_eye_orientation(0)
        self.get_logger().info('Requesting first quaternion to compute planning')

        # Dovrai cercare posizione del frame                                                            da guardare
        q_end = (-0.15, 0.45, 0.32538, w, x, y, z)

        # Compute trajectory, quintic polynomial
        self.trajectory = mtraj(quintic, self.current_pose, q_end, STEP_MAX)
                               
    def move_first(self):
        step = 0
        
        while step < STEP_MAX :
            q = self.trajectory.q[step]

            position = (q[0], q[1], q[2])
            orientation = (q[3], q[4], q[5], q[6])

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)      
            step += 1
    
    def move_second(self):
        while True:
            w, x, y, z  = self.get_eye_orientation(0)

            position = (self.current_pose[0], self.current_pose[1], self.current_pose[2])
            orientation = (w, x, y, z)

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)        
    
    def get_eye_orientation(self, request):
        self.req.r = request
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        q_eye = np.array([response.w, -response.x, response.y, -response.z])
        q_trans = euler.euler2quat(0, 0, -np.pi/2, 'rzyx')
        q = self.quaternion_multiply(q_trans, q_eye)

        return q
    
    def publish_pose(self, position, orientation):
        msg = PoseStamped()

        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]

        msg.pose.orientation.w = orientation[0]
        msg.pose.orientation.x = orientation[1]
        msg.pose.orientation.y = orientation[2]
        msg.pose.orientation.z = orientation[3]
        
        self.current_pose = np.concatenate((position, orientation))
        self.publisher.publish(msg)
        self.get_logger().info('Publishing pose')

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    


def main():
    rclpy.init()

    ur3_controller = Ur3_controller()
    rclpy.spin(ur3_controller)
    ur3_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
