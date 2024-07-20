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

# Multiply time in seconds
APPROACHING_STEPS = FREQUENCY * 3
DEMONSTRATION_STEPS = FREQUENCY * 5
FINDING_INJECTION_POSITION_STEPS = FREQUENCY * 3
INJECTION_STEPS = FREQUENCY * 2
SUBSEQUENCE_STEPS = int(FREQUENCY * 0.1)

DATASET_NUMBER_STD_DEVIATION = 20
STABILITY_THRESHOLD = 0.01

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
        self.current_pose = self.get_initial_pose()

        # Setting target position
        self.target_position = np.array([0.30, 0.35, 0.35])
#        self.target_position = np.array([0.25, 0.30, 0.30])
#        self.target_position = np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2]])
                                
        # Setting safe distance from the center of the eye
        self.safe_distance = 0.05
#        self.safe_distance = 0.0
        
        # Setting injection angle
        self.injection_angle = np.pi/3

        # Creating buffer for deviation standard computing
        self.variation_buffer = []

        # Publishing frame with safe distance included, (USED IN SIMULATION)
        self.publish_RCM_frame()
        
        # Complete motion execution
        while True:
            self.get_logger().info('Target approaching')
            self.approaching_target_position()

            #self.get_logger().info('Following eye as a demonstration')
            #self.eye_following()

            self.get_logger().info('Finding injection angle')
            self.finding_injection_angle()
            
            self.get_logger().info('Performing injection')
            self.eye_injection()

            #self.get_logger().info('Returning to safe position')
            #self.approaching_target_position()

            self.get_logger().info('Performance finished')
            time.sleep(5)

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
                return  np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
                                    trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z])
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
        position = self.traslate_position_from_RCM(self.target_position, orientation, self.safe_distance)
        
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
            position = self.traslate_position_from_RCM(self.target_position, orientation, self.safe_distance)

            # Computing trajectory
            arriving_pose = np.concatenate((position, orientation))
            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)

            # Publishing the trajectory
            self.publish_trajectory()

            step += SUBSEQUENCE_STEPS
        
    def finding_injection_angle(self):
        step = SUBSEQUENCE_STEPS
        while step < FINDING_INJECTION_POSITION_STEPS:
            # Getting current eye orientation
            orientation  = self.get_eye_orientation(0)

            # Finding progressive angle from current position
            angle_to_center = self.get_angle_from_center(orientation)
            angle = (self.injection_angle - angle_to_center)*(step/FINDING_INJECTION_POSITION_STEPS) + angle_to_center

            # Finding arriving orientation
            orientation = self.get_injection_orientation(orientation, angle)
            orientation  = self.rotating_orientation_toward_target(orientation)

            # Finding arring position, adding safe distance 
            position = self.traslate_position_from_RCM(self.target_position, orientation, self.safe_distance)
            
            # Computing trajectory
            arriving_pose = np.concatenate((position, orientation))
            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)

            # Publishing the trajectory
            self.publish_trajectory()

            step += SUBSEQUENCE_STEPS
    
    def eye_injection(self):
        step = 0
        while step < INJECTION_STEPS:
            # Finding arriving orientation
            orientation = self.get_eye_orientation(0)
            orientation = self.get_injection_orientation(orientation, self.injection_angle)
            orientation  = self.rotating_orientation_toward_target(orientation)

            # Checking if the eye is stable, if it is not restart
            if self.stability_check == False:
                step = 0

            # Slowly reducing the distance from RCM, performig injection
            distance = self.safe_distance*(1.0 - step/INJECTION_STEPS)

            # Finding arring position, adding progressive distance
            position = self.traslate_position_from_RCM(self.target_position, orientation, distance)

            # Computing trajectory
            arriving_pose = np.concatenate((position, orientation))
            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)

            # Publishing the trajectory
            self.publish_trajectory()

            step += SUBSEQUENCE_STEPS

        step = SUBSEQUENCE_STEPS
        while step < INJECTION_STEPS:
            distance = self.safe_distance*(step/INJECTION_STEPS)

            orientation = self.get_eye_orientation(0)
            orientation = self.get_injection_orientation(orientation, self.injection_angle)
            orientation  = self.rotating_orientation_toward_target(orientation)

            position = self.traslate_position_from_RCM(self.target_position, orientation, distance)

            arriving_pose = np.concatenate((position, orientation))

            self.trajectory_planning(self.current_pose, arriving_pose, SUBSEQUENCE_STEPS)
            self.publish_trajectory()
            step += SUBSEQUENCE_STEPS

    def get_injection_orientation(self, orientation, angle):
        # This function find out the new quaternion from the given one
        # rotating it, of the angle given, around the RCM
        # passing through the center of the eye, represented by the quaternion q = (w: 1.0, x: 0.0, y: 0.0, z: 0.0)
    
        roll, pitch, yaw = euler.quat2euler(orientation, 'rzyx')

        # First control the special cases yaw = 0 and pitch = 0, then computing the general case
        if yaw == 0:
            roll = 0
            pitch = - np.sign(pitch)*angle
        elif pitch == 0:
            roll = 0
            yaw = - np.sign(yaw)*angle
        else:  
            roll = - np.arctan2(np.sin(yaw/2), np.sin(pitch/2))

            # Tangent of 135° and -45° are equal, but that is not true for the frame rotation
            if np.abs(roll) > np.pi/2:
                roll = - np.sign(roll)*(np.pi - np.abs(roll))

            pitch = - np.sign(pitch)*angle
            yaw = 0
        
        q = euler.euler2quat(roll, pitch, yaw, 'rzyx')
        q = self.quaternion_multiply(orientation, q)
        return q

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
    
    def stability_control(self, orientation):
        self.stability_check = False
        lenght = len(self.variation_buffer)

        if lenght == 0:
            self.variation_buffer = np.array([orientation])
        elif lenght < DATASET_NUMBER_STD_DEVIATION:
            self.variation_buffer = np.append(self.variation_buffer, [orientation], axis=0)
        else:
            self.variation_buffer = np.delete(self.variation_buffer, 0, axis=0)
            self.variation_buffer = np.append(self.variation_buffer, [orientation], axis=0)

            std = np.std(self.variation_buffer, ddof=1, axis=0)
            std_mean = np.mean(std)
            #print(std_mean)
            if std_mean < STABILITY_THRESHOLD:
                self.stability_check = True

    def get_eye_orientation(self, request):
        # Sending request to the server
        self.req.r = request
        future = self.cli.call_async(self.req)

        # Waiting for the response
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        # Saving response in an array 
        q = np.array([response.w, response.x, response.y, response.z])
        self.stability_control(q)

        return q

    def get_angle_from_center(self, orientation):
        roll, pitch, yaw = euler.quat2euler(orientation, 'rzyx')

        angle = 2 * np.arcsin(np.sqrt(1 - (np.cos(pitch) + np.cos(yaw))/2))

        return np.abs(angle)
    
    def rotating_orientation_toward_target(self, orientation):
        # Mirroring the quaternion
        q = np.array([orientation[0], -orientation[1], orientation[2], -orientation[3]])
        
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
        #w0, x0, y0, z0 = quaternion0
        #w1, x1, y1, z1 = quaternion1
        #q_f = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        #                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        #                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        #                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
        
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
