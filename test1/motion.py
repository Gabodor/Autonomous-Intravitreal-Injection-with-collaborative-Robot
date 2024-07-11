#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from project_interfaces.srv import Pose
from geometry_msgs.msg import PoseStamped, TransformStamped

from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

from roboticstoolbox import quintic, mtraj
import time
import numpy as np
from transforms3d import euler

# Constant values
STEP_MAX = 80
FREQUENCY = 20
TIME_STEP = 1/FREQUENCY
TIME_MAX_SEC = STEP_MAX/FREQUENCY

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

        # Setting target position
        self.end_pose = (-0.15, 0.40, 0.35)
        orientation = self.transform_orientation_to_eye(np.array([1.0, 0.0, 0.0, 0.0]))
        self.end_pose = (self.end_pose[0], self.end_pose[1], self.end_pose[2],  orientation[0], orientation[1], orientation[2], orientation[3])

        # Setting safe distance from eye
        self.safe_distance = 0.05
        
        # Setting injection angle
        self.injection_angle = np.pi/2

        # Setting the timer for the demonstration motion, eye_following, expressed in seconds
        self.demonstration_time = 5
        
        # Getting current configuration 
        self.get_initial_pose()

        # Publishing frame with safe distance included
        self.publish_static_tranform()

        # Computing trajectory, saving it in self.trajectory
        self.trajectory_planning()
        
        # Complete motion execution
        while True:
            self.eye_approaching()
            self.eye_following()
            self.eye_injection()

            self.get_logger().info('Performance finished')
            time.sleep(5)

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
    
    def trajectory_planning(self):
        self.get_logger().info('Computing trajectory')
        self.trajectory = mtraj(quintic, self.current_pose, self.end_pose, STEP_MAX)
                               
    def eye_approaching(self):
        step = 0
        self.get_logger().info('Target approaching')

        while step < STEP_MAX :
            pose = self.trajectory.q[step]

            position = (pose[0], pose[1], pose[2])
            orientation = (pose[3], pose[4], pose[5], pose[6])

            position = self.safe_distance_position(position, orientation, self.safe_distance)

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)      
            step += 1
    
    def eye_following(self):
        step = 0
        self.get_logger().info('Following eye as a demonstration')

        while step < FREQUENCY*self.demonstration_time:
            orientation  = self.get_eye_orientation(0)
            orientation  = self.transform_orientation_to_eye(orientation)

            position = np.array([self.end_pose[0], self.end_pose[1], self.end_pose[2]])

            position = self.safe_distance_position(position, orientation, self.safe_distance)

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)
            step += 1
        
    def eye_injection(self):
        step = 0
        self.get_logger().info('Finding injection angle')

        while step < STEP_MAX:
            q  = self.get_eye_orientation(0)
            position = np.array([self.end_pose[0], self.end_pose[1], self.end_pose[2]])

            angle = (self.injection_angle)*(step/STEP_MAX)
            orientation = self.get_injection_orientation(q, angle)
            orientation  = self.transform_orientation_to_eye(orientation)


            position = self.safe_distance_position(position, orientation, self.safe_distance)

            self.publish_pose(position, orientation)
            time.sleep(TIME_STEP)
            step += 1

        step = 0
        self.get_logger().info('Performing injection')

        while step < STEP_MAX:
            position = np.array([self.end_pose[0], self.end_pose[1], self.end_pose[2]])

            q = self.get_eye_orientation(0)
            orientation = self.get_injection_orientation(q, self.injection_angle)
            orientation  = self.transform_orientation_to_eye(orientation)

            distance = self.safe_distance*(1.0 - step/STEP_MAX)
            position = self.safe_distance_position(position, orientation, distance)

            self.publish_pose(position, orientation)
            time.sleep(TIME_STEP)
            step += 1

    def get_injection_orientation(self, orientation, angle):
        roll, pitch, yaw = euler.quat2euler(orientation, 'rzyx')

        if yaw == 0:
            roll = 0
            pitch = - np.sign(pitch)*angle
        elif pitch == 0:
            roll = 0
            yaw = - np.sign(yaw)*angle
        else:  
            roll = - np.arctan2(np.sin(yaw/2), np.sin(pitch/2))
            # tan of 135 or -45 degree is the same thing but not for the frame rotation
            if np.abs(roll) > np.pi/2:
                roll = - np.sign(roll)*(np.pi - np.abs(roll))
            pitch = - np.sign(pitch)*angle
            yaw = 0
        
        q = euler.euler2quat(roll, pitch, yaw, 'rzyx')
        q = self.quaternion_multiply(orientation, q)
        return q

    def safe_distance_position(self, position, orientation, safe_distance):
        q = orientation
        q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

        p = np.array([0, position[0], position[1], position[2]])

        traslation = self.quaternion_multiply(q, np.array([0, 0, 0, -safe_distance]))
        traslation = self.quaternion_multiply(traslation, q_conjugate)

        p_traslated = np.array([p[1] + traslation[1], p[2] + traslation[2], p[3] + traslation[3]])
        return p_traslated
    
    def get_eye_orientation(self, request):
        self.req.r = request
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        q = np.array([response.w, response.x, response.y, response.z])

        return q
    
    def transform_orientation_to_eye(self, orientation):
        # Mirroring the quaternion
        q = np.array([orientation[0], -orientation[1], orientation[2], -orientation[3]])
        
        # Rotating the end-effector towards the end_pose
        alpha = np.arctan2(self.end_pose[1], self.end_pose[0]) - np.pi/2
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
        
        #self.current_pose = np.concatenate((position, orientation))
        self.publisher.publish(msg)
        #self.get_logger().info('Publishing pose')

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
