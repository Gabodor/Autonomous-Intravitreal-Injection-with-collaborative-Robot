#!/usr/bin/env python3
from project_interfaces.srv import Pose

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import argparse
import numpy as np
import cv2
import time
import threading
from queue import Queue
from collections import deque

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import math
from transforms3d import euler

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render, getDataset

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from visualization_msgs.msg import  Marker

ROLL = 0.0
PITCH = 0.0
YAW = 0.0

PUBLISH_FREQUENCY = 0.2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def eye_tracking():
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        #weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        weights = getDataset(),
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
    # checking camera index: 'ls -al /dev/video*'
    # adding '--cam N_CAMERA' to command line to change camera
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    global ROLL 
    global PITCH
    global YAW

    with torch.no_grad():
        while True:
            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = gaze_pipeline.step(frame)
            ROLL = 0.0
            PITCH = results.pitch[0]
            YAW = results.yaw[0]

            # Visualize output
            frame = render(frame, results)
           
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        # Create buffer for smoothing the orientation
        self.buffer = deque()
        self.buffer.append(np.array([ROLL,PITCH,YAW]))

        self.srv = self.create_service(Pose, 'test1', self.service_callback)

        # Create publisher to publish eye movement in the simulation 
        self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(PUBLISH_FREQUENCY, self.routine)

    def routine(self):
        if len(self.buffer) < 8:
            orientation = np.array([ROLL, PITCH, YAW])
            for i in range(3):
                orientation[i] = self.angle_limit_control(orientation[i], 0.70)

            last_orientation = self.buffer[-1]
            diff = (orientation - last_orientation)/3

            for i in range(3):
                last_orientation += diff
                self.buffer.append(last_orientation)

        self.environment_building()

    def angle_limit_control(self, angle, limit):
        if angle > limit:
            angle = limit
        if angle < -limit:
            angle = -limit
        return angle

    def get_quaternion(self):
        #print(len(self.buffer))
        if len(self.buffer) > 1:
            roll, pitch, yaw = self.buffer.popleft()
        else:
            roll, pitch, yaw = self.buffer[0]

        #roll, pitch, yaw = 0,  -0.785,  -0.785

        w, x, y, z = euler.euler2quat(roll, pitch, yaw, 'rzyx')

        #print("[roll: %f, pitch: %f, yaw: %f]" % (roll, pitch, yaw))
        #print("[w: %f, x: %f, y: %f, z: %f]" % (w, x, y, z))    

        return w, x, y, z

    def service_callback(self, request, response):
        response.w, response.x, response.y, response.z = self.get_quaternion()
        self.get_logger().info('Incoming request r: %d' % (request.r))

        return response

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    
    def environment_building(self):     
        roll, pitch, yaw = self.buffer[0]
        q = euler.euler2quat(roll, pitch, yaw, 'rzyx')  
        q_trans = euler.euler2quat(np.pi, 0, -np.pi/2, 'rzyx')
        w, x, y, z = self.quaternion_multiply(q_trans, q)

        size = 1
   
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'eye_frame'
        t.transform.translation.x = -0.15
        t.transform.translation.y = 0.40
        t.transform.translation.z = 0.32538
        t.transform.rotation.w = w
        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z

        self.tf_broadcaster.sendTransform(t)

        marker = Marker()

        marker.header.frame_id = "eye_frame"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.scale.x = 0.05 * size
        marker.scale.y = 0.05 * size
        marker.scale.z = 0.05 * size
        marker.color.a = 1.0
        marker.color.r = 255.0 / 255.0
        marker.color.g = 255.0 / 255.0
        marker.color.b = 255.0 / 255.0

        self.marker_publisher.publish(marker)

        marker.header.frame_id = "eye_frame"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "my_namespace"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.02 * size
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.scale.x = 0.025 * size
        marker.scale.y = 0.025 * size
        marker.scale.z = 0.01 * size
        marker.color.a = 1.0
        marker.color.r = 0.0 / 255.0
        marker.color.g = 150.0 / 255.0
        marker.color.b = 0.0 / 255.0

        self.marker_publisher.publish(marker)

        marker.header.frame_id = "eye_frame"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "my_namespace"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.025 * size
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.scale.x = 0.01 * size
        marker.scale.y = 0.01 * size
        marker.scale.z = 0.0025 * size
        marker.color.a = 1.0
        marker.color.r = 0.0 / 255.0
        marker.color.g = 0.0 / 255.0
        marker.color.b = 0.0 / 255.0

        self.marker_publisher.publish(marker)

def main(args=None):
    t1 = threading.Thread(target=eye_tracking,args=())
    t1.start()

    rclpy.init()
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)

    t1.join()
    rclpy.shutdown()

if __name__ == '__main__':
    main()