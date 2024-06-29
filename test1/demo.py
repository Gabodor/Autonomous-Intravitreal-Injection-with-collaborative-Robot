#!/usr/bin/env python3
from project_interfaces.srv import Pose

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import argparse
import numpy as np
import cv2
import time

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

class MinimalService(Node):

    def __init__(self, cap, gaze_pipeline):
        super().__init__('minimal_service')
        self.cap = cap
        self.gaze_pipeline = gaze_pipeline
        self.srv = self.create_service(Pose, 'test1', self.test1_callback)

    def RPY_to_quaternion(self, roll, pitch, yaw):
        w, x, y, z = euler.euler2quat(roll, pitch, yaw, 'rzyx')

        print("[roll: %f, pitch: %f, yaw: %f]" % (roll,pitch,yaw))
        print("[w: %f, x: %f, y: %f, z: %f]" % (w, x, y, z))

        return w, x, y, z

    def get_quaternion(self):
        print("wow1")
        with torch.no_grad():
            # Get frame
            print("wow1")
            success, frame = self.cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = self.gaze_pipeline.step(frame)
            w, x, y, z = self.RPY_to_quaternion(0, results.pitch[0], results.yaw[0])
            #w, x, y, z = self.RPY_to_quaternion(0, 0, 0)

            # Visualize output
            frame = render(frame, results)
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                success,frame = self.cap.read()  
            
            return w, x, y, z

    def test1_callback(self, request, response):
        response.w, response.x, response.y, response.z = self.get_quaternion()
        self.get_logger().info('Incoming request\n r: %d' % (request.r))

        return response

def main(args=None):
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
     
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

        
    rclpy.init()

    minimal_service = MinimalService(cap, gaze_pipeline)

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

            # Visualize output
            frame = render(frame, results)
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()