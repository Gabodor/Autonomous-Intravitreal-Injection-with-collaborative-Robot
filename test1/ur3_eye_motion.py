#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from project_interfaces.srv import Pose
from geometry_msgs.msg import PoseStamped, TransformStamped

from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

from roboticstoolbox import quintic, mtraj, ctraj
from spatialmath import SE3, UnitQuaternion
import time
import numpy as np
from transforms3d import euler
import pyquaternion as pyq

# Constant values
FREQUENCY = 500
TIME_STEP = 1/FREQUENCY

APPROACHING_STEPS = FREQUENCY * 5
DEMONSTRATION_STEPS = FREQUENCY * 20
FINDING_INJECTION_POSITION_STEPS = FREQUENCY * 5
INJECTION_STEPS = FREQUENCY * 2
SUBSEQUENCE_STEPS = int(FREQUENCY * 0.1)

class Ur3_controller(Node):
    def __init__(self):
        # Initializing class
        super().__init__('ur3_controller')
        
        # Creating client(eye-tracking) to get pose
        self.cli = self.create_client(Pose, 'test1')

        # Waiting for eye-tracking to start
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Pose.Request()

        # Creating publisher to publish planning
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 10)

        # Getting current configuration 
        self.get_initial_pose()

        # Setting target position
#        self.end_pose = (-0.1, 0.1, 0.35)
        self.end_pose = (-0.15, 0.40, 0.35)
#        self.end_pose = (self.current_pose[0], self.current_pose[1], self.current_pose[2])
        orientation = self.transform_orientation_to_eye(np.array([1.0, 0.0, 0.0, 0.0]))
        self.end_pose = (self.end_pose[0], self.end_pose[1], self.end_pose[2],  orientation[0], orientation[1], orientation[2], orientation[3])
        
        # Setting safe distance from eye
        self.safe_distance = 0.05

        # Publishing frame with safe distance included, (USED IN SIMULATION)
        self.publish_static_tranform()

        # Complete motion execution
        self.eye_location_approaching()
        self.eye_following()

    def publish_static_tranform(self):
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'wrist_3_link'
        t.child_frame_id = 'safe_distance'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.safe_distance

        t.transform.rotation.w = 1.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0

        self.tf_static_broadcaster.sendTransform(t)

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
                self.get_logger().info(f'Could not transform: {ex}')
    
    def trajectory_planning(self, initial_pose, final_pose, steps):
        #self.get_logger().info('Computing trajectory')

        position = [initial_pose[0], initial_pose[1], initial_pose[2]]
        orientation = [initial_pose[3], initial_pose[4], initial_pose[5], initial_pose[6]]
        T1 = SE3.Rt(UnitQuaternion(orientation).R, position)

        position = [final_pose[0], final_pose[1], final_pose[2]]
        orientation = [final_pose[3], final_pose[4], final_pose[5], final_pose[6]]
        T2 = SE3.Rt(UnitQuaternion(orientation).R, position)

        self.trajectory = []
        tmp = ctraj(T1, T2, steps)

        for i in range(len(tmp)):
            quat = pyq.Quaternion(matrix=tmp[i].R).normalised
            self.trajectory.append([tmp[i].t[0], tmp[i].t[1], tmp[i].t[2], quat.w, quat.x, quat.y, quat.z])
                               
    def eye_location_approaching(self):
        self.get_logger().info('Target approaching')

        # Computing trajectory, saving it in self.trajectory
        orientation = self.transform_orientation_to_eye(np.array([1.0, 0.0, 0.0, 0.0]))
        position = np.array([self.end_pose[0], self.end_pose[1], self.end_pose[2]])
        position = self.safe_distance_position(position, orientation, self.safe_distance)


        arriving_pose = np.concatenate((position, orientation))
        self.trajectory_planning(self.current_pose, arriving_pose, APPROACHING_STEPS)

        self.publish_trajectory()
    
    def eye_following(self):
        self.get_logger().info('Following eye')

        while True:
            # Computing trajectory
            orientation  = self.get_eye_orientation(0)
            orientation  = self.transform_orientation_to_eye(orientation)

            position = np.array([self.end_pose[0], self.end_pose[1], self.end_pose[2]])
            position = self.safe_distance_position(position, orientation, self.safe_distance)


            arriving_pose = np.concatenate((position, orientation))

            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)
            self.publish_trajectory()
    
    def get_eye_orientation(self, request):
        self.req.r = request
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        q = np.array([response.w, response.x, response.y, response.z])

        return q
    
    def safe_distance_position(self, position, orientation, safe_distance):
        q = orientation
        q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

        p = np.array([0, position[0], position[1], position[2]])

        traslation = self.quaternion_multiply(q, np.array([0, 0, 0, -safe_distance]))
        traslation = self.quaternion_multiply(traslation, q_conjugate)

        p_traslated = np.array([p[1] + traslation[1], p[2] + traslation[2], p[3] + traslation[3]])
        return p_traslated
    
    def transform_orientation_to_eye(self, orientation):
        # Mirroring the quaternion
        
        # Rotating the end-effector towards the end_pose
        alpha = np.arctan2(self.end_pose[1], self.end_pose[0]) - np.pi/2        # + np.pi
        q_trans = euler.euler2quat(alpha, 0, -np.pi/2, 'rzyx')
        orientation = self.quaternion_multiply(q_trans, orientation)
        return orientation
    
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
        #self.get_logger().info('Publishing pose')
    
    def publish_trajectory(self):
        step = 1
        while step < len(self.trajectory):
            pose = self.trajectory[step]

            position = (pose[0], pose[1], pose[2])
            orientation = (pose[3], pose[4], pose[5], pose[6])

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)     
            step += 1

    def quaternion_multiply(self, quaternion1, quaternion0):
        # Questa funzione deve sparire, trova libreria che la implementa
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
