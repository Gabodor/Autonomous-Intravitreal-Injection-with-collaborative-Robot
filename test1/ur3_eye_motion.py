#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from project_interfaces.srv import Pose
from geometry_msgs.msg import PoseStamped, TransformStamped

from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

from roboticstoolbox import ctraj, mtraj, quintic
from spatialmath import SE3, UnitQuaternion
import time
import numpy as np
from transforms3d import euler
import pyquaternion as pyq

# Constant values
TIME_DURATION = 5
FREQUENCY = 500
STEP_MAX = TIME_DURATION * FREQUENCY
TIME_STEP = 1/FREQUENCY
TIME_MAX_SEC = STEP_MAX/FREQUENCY

class Ur3_controller(Node):
    def __init__(self):
        # Initializing class
        super().__init__('ur3_controller')

        # Creating publisher to publish planning
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 10)

#        pose = (-0.15, 0.40, 0.35, 1.0, 0.0, 0.0, 0.0)
#        position = (pose[0], pose[1], pose[2])
#        orientation = (pose[3], pose[4], pose[5], pose[6])
#        self.publish_pose(position, orientation)

        # Getting current configuration 
        print("initial pose searching")
        self.get_initial_pose()
        print("initial pose received")
        print(self.current_pose)

        #self.current_pose
#        self.current_pose = (-0.15, 0.40, 0.35, 1.0, 0.0, 0.0, 0.0)

        # Setting target position
#        self.end_pose = (-0.15, 0.40, 0.35, 1.0, 0.0, 0.0, 0.0)
        self.end_pose = (self.current_pose[0] + 0.05, self.current_pose[1] + 0.05, self.current_pose[2] + 0.05, self.current_pose[3], self.current_pose[4], self.current_pose[5], self.current_pose[6])

        # Computing trajectory, saving it in self.trajectory
        print("initial pose searching")
        self.trajectory_planning()
        print("initial pose searching")
        
        # Complete motion execution
        self.eye_approaching()

        self.get_logger().info('Performance finished')

    def get_initial_pose(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        waiting_pose = True
        while waiting_pose:           
            try:
                future = self.tf_buffer.wait_for_transform_async(
                        'base_link',
                        'tool0',
                        rclpy.time.Time()
                        )
                rclpy.spin_until_future_complete(self, future)
                trans = self.tf_buffer.lookup_transform(
                    'base_link',
                    'tool0',
                    rclpy.time.Time()
                    )
                self.current_pose = (   trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
                                        trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z)
                waiting_pose = False
            except TransformException as ex:
                self.get_logger().info(f'Could not transform: {ex}')
    
    def trajectory_planning(self):
        self.get_logger().info('Computing trajectory')

        position = [self.current_pose[0], self.current_pose[1], self.current_pose[2]]
        orientation = [self.current_pose[3], self.current_pose[4], self.current_pose[5], self.current_pose[6]]
        T1 = SE3.Rt(UnitQuaternion(orientation).R, position)

        position = [self.end_pose[0], self.end_pose[1], self.end_pose[2]]
        orientation = [self.end_pose[3], self.end_pose[4], self.end_pose[5], self.end_pose[6]]
        T2 = SE3.Rt(UnitQuaternion(orientation).R, position)

        self.trajectory = []
        tmp = ctraj(T1, T2, STEP_MAX)
        for i in range(len(tmp)):
            quat = pyq.Quaternion(matrix=tmp[i].R).normalised
            self.trajectory.append([tmp[i].t[0], tmp[i].t[1], tmp[i].t[2], quat.w, quat.x, quat.y, quat.z])

        #self.trajectory = mtraj(quintic, self.current_pose, self.end_pose, STEP_MAX)
                               
    def eye_approaching(self):
        step = 0
        self.get_logger().info('Target approaching')

        while step < STEP_MAX :
            pose = self.trajectory[step]

            position = (pose[0], pose[1], pose[2])
            orientation = (pose[3], pose[4], pose[5], pose[6])
            #print(position)

            self.publish_pose(position, orientation)

            time.sleep(TIME_STEP)      
            step += 1
    
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
    
def main():
    rclpy.init()

    ur3_controller = Ur3_controller()
    rclpy.spin(ur3_controller)

    ur3_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
