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
#FREQUENCY = 80
TIME_STEP = 1/FREQUENCY

# Multiply time in seconds
APPROACHING_STEPS = FREQUENCY * 3 #5
DEMONSTRATION_STEPS = FREQUENCY * 5 #20
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
#        self.target_position = (-0.15, 0.35, 0.35)
        self.target_position = (0.30, 0.35, 0.35)
#        self.target_position = (0.25, 0.30, 0.30)
#        self.target_position = (self.current_pose[0], self.current_pose[1], self.current_pose[2])

        orientation = self.rotating_orientation_toward_target(np.array([1.0, 0.0, 0.0, 0.0]))
        self.target_position = (self.target_position[0], self.target_position[1], self.target_position[2],  orientation[0], orientation[1], orientation[2], orientation[3])

        # Setting safe distance from the center of the eye
        self.safe_distance = 0.05
#        self.safe_distance = 0.0

        # Publishing frame with safe distance included, (USED IN SIMULATION)
        self.publish_RCM_frame()
        
        # Complete motion execution
        self.get_logger().info('Target approaching')
        self.approaching_target_position()

        self.get_logger().info('Simulating eye movement')
        self.eye_following()

        self.get_logger().info('Performance finished')

    def publish_RCM_frame(self):
        # RCM is the Remote Center of Motion
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'wrist_3_link'
        t.child_frame_id = 'RCM_frame'

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

        # Waiting to read the current pose 
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

        # Computing homogeneous transformation matrix of initial position
        position = [initial_pose[0], initial_pose[1], initial_pose[2]]
        orientation = [initial_pose[3], initial_pose[4], initial_pose[5], initial_pose[6]]
        T1 = SE3.Rt(UnitQuaternion(orientation).R, position)

        # Computing homogeneous transformation matrix of final position
        position = [final_pose[0], final_pose[1], final_pose[2]]
        orientation = [final_pose[3], final_pose[4], final_pose[5], final_pose[6]]
        T2 = SE3.Rt(UnitQuaternion(orientation).R, position)

        self.trajectory = []
        tmp = ctraj(T1, T2, steps)

        # Exporting trajectory as a list of position
        for i in range(len(tmp)):
            quat = pyq.Quaternion(matrix=tmp[i].R).normalised
            self.trajectory.append([tmp[i].t[0], tmp[i].t[1], tmp[i].t[2], quat.w, quat.x, quat.y, quat.z])
                               
    def approaching_target_position(self):
        # Target neutral orientation
        orientation = self.rotating_orientation_toward_target(np.array([1.0, 0.0, 0.0, 0.0]))

        # Target position adding safe distance
        position = np.array([self.target_position[0], self.target_position[1], self.target_position[2]])
        position = self.traslate_position_from_RCM(position, orientation, self.safe_distance)
        
        # Computing trajectory
        arriving_pose = np.concatenate((position, orientation))
        self.trajectory_planning(self.current_pose, arriving_pose, APPROACHING_STEPS)

        # Publishing the trajectory
        self.publish_trajectory()
    
    def eye_following(self):
        step = 0
        while step < DEMONSTRATION_STEPS:
            # Finding arriving orientation
            orientation  = self.get_eye_orientation(0)
            orientation  = self.rotating_orientation_toward_target(orientation)

            # Finding arring position, adding safe distance 
            position = np.array([self.target_position[0], self.target_position[1], self.target_position[2]])
            position = self.traslate_position_from_RCM(position, orientation, self.safe_distance)

            # Computing trajectory
            arriving_pose = np.concatenate((position, orientation))
            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)

            # Publishing the trajectory
            self.publish_trajectory()

            step += SUBSEQUENCE_STEPS

    def traslate_position_from_RCM(self, position, orientation, traslation):
        # Finding the conjugate of the orientation, to compute transformation
        q = orientation
        q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

        # Normalizing traslation, in order to use quaternions
        # Computing traslation oriented as orientation, q * traslation * q_conjugate
        traslation = self.quaternion_multiply(q, np.array([0, 0, 0, -traslation]))
        traslation = self.quaternion_multiply(traslation, q_conjugate)

        # Adding traslation
        p_traslated = np.array([position[0] + traslation[1], position[1] + traslation[2], position[2] + traslation[3]])
        return p_traslated
    
    def get_eye_orientation(self, request):
        # Sending request to the server
        self.req.r = request
        future = self.cli.call_async(self.req)

        # Waiting for the response
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        # Saving response in an array 
        q = np.array([response.w, response.x, response.y, response.z])

        return q
    
    def rotating_orientation_toward_target(self, orientation):
        # Rotating the end-effector towards the target_position
        alpha = np.arctan2(self.target_position[1], self.target_position[0]) - np.pi/2
        q_trans = euler.euler2quat(alpha, 0, -np.pi/2, 'rzyx')
        q = self.quaternion_multiply(q_trans, q)

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
        q1 = pyq.Quaternion(quaternion1)
        q0 = pyq.Quaternion(quaternion0)

        q = q1*q0
        return np.array([q.w, q.x, q.y, q.z])
    
def main():
    rclpy.init()

    ur3_controller = Ur3_controller()
    rclpy.spin(ur3_controller)

    ur3_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
