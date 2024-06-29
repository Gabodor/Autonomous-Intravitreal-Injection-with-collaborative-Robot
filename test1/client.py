#!/usr/bin/env python3
import sys
from project_interfaces.srv import Pose

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage

from roboticstoolbox import quintic, mtraj
import numpy as np
import quaternion
from transforms3d import euler

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
        response = future.result()
        
        q_trans = euler.euler2quat(-np.pi/2, 0, -np.pi/2, 'rzyx')
        q_eye = np.array([response.w, response.x, response.y, response.z])
        q = self.quaternion_multiply(q_trans, q_eye)

        return q
    
    def trajectory_planning(self):                                                          #da rivedere
        # Deve chiedere configurazione attuale end-effector e finale
        response = self.send_request(0)
        self.get_logger().info('Requesting first quaternion to compute planning')

        while not self.check:
            pass

        q_end=(0.37158, 0.15214, 0.32538, response[1], response[2], response[3], response[0])
        self.trajectory = mtraj(quintic, self.current_pose, q_end, STEP_MAX)
    
    def publish_pose(self, position, orientation):
        msg = PoseStamped()

        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        
        self.current_pose = np.concatenate((position, orientation))
        self.publisher.publish(msg)
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
    
    def move_second(self):
        while True:
            response = self.send_request(0)
            position = (self.current_pose[0], self.current_pose[1], self.current_pose[2])
            orientation = (response[1], response[2], response[3], response[0])

            self.get_logger().info('Requesting quaternion')
            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)        

    def listener_callback(self, msg):
        self.destroy_subscription(self.subscription)

        translation = np.array([])
        rotation = np.array([])

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
            np.append(translation, [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            np.append(rotation, [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

        tmp = translation[0]
        translation[0] = translation[2]
        translation[2] = tmp

        tmp = rotation[0]
        rotation[0] = rotation[2]
        rotation[2] = tmp

        position = [0.0, 0.0, 0.0]

        q = translation[0]
        for i in range(0, 6):
            if i!=0 :
                q = self.quaternion_multiply(translation[i], q)
            q_c = quaternion.conj
        
        self.current_pose = (-0.13088, 0.29877, 0.30323, 0.99999, 0.00039, 0.00039, 0.00039)
        self.check = True

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
    print("arrivato?")
    rclpy.spin(ur3_controller)

    ur3_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
