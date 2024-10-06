# Autonomous Intravitreal Injection with collaborative Robot
This repository provide a ROS2 package, which implement the trajectory planning for an UR3e performing an autonomous injection, following the eye motion in real-time.

## Installation

### Setup
- Ubuntu 22.04
- [ROS2 Humble](https://docs.ros.org/en/humble/index.html)

### L2CS-Net
- Official git project [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)
- It only need a common RGB camera, a webcam
#### 1. Clone the repository:
```
cd 
git clone https://github.com/Ahmednull/L2CS-Net
```
#### 2. Add dataset path finding function:
```
cd ~/L2CS/l2cs
touch dataset_path.py
```
- Copy following lines in the dataset_path.py file
```
import pathlib

def getDataset():
    CWD = pathlib.Path.cwd() / 'models' / 'L2CSNet_gaze360.pkl'
    return CWD
```
#### 3. Modify init.py file:
```
cd ~/L2CS/l2cs
```
Open __init__.py and append the following lines:
- Append under the other import:
```
from dataset_path import getDataset
```
- Append inside the square bracket as last item:
```
'getDataset'
```
#### 3. Download dataset:
```
cd ~/L2CS/l2cs
mkdir models
cd models
```
- Download Gaze360 dataset from [here](http://gaze360.csail.mit.edu/download.php) inside this directory
- Otherwise you can check [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) other dataset proposal

#### 4. Create python library:
```
cd ~/L2CS/l2cs
pip install [-e] .
```
Now you should be able to import the package with the following command:
```
$ python
>>> import l2cs
```

### Prerequisites
#### 1. Install required packages:
ROS 2 Dependencies
```
sudo apt update
sudo apt install xsltproc ros-humble-map-msgs ros-humble-pendulum-msgs ros-humble-example-interfaces ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-hardware-interface
```  
#### 2. If you do not have one, create a new ROS2 workspace:
```
mkdir -p ~/ros2_ws/src
```
#### 3. Clone this repo in your workspace src folder:  
```
cd ~/ros2_ws/src
git clone https://github.com/Gabodor/Autonomous-Intravitreal-Injection-with-collaborative-Robot.git ur3_injection_controller
```
#### 4. Create the interface used in the project:  
- Create a new ROS2 package:
```
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake --license Apache-2.0 project_interfaces
cd project_interfaces
```
- Open CMakeLists.txt and append the following lines:
```
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Pose.srv"
)
```
- Open package.xml and append the following lines:
```
<buildtool_depend>rosidl_default_generators</buildtool_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```
- Create the service:
```
mkdir srv
cd srv
touch Pose.srv
```
- Open Pose.srv file and copy inside the following text:
```
int64 r
---
float64 x
float64 y
float64 z
float64 w
```
#### 5. Build workspace:  
```
cd $HOME/ros2_ws/
colcon build --symlink-install
```
#### 6. Controllers
In order to use this ROS2 package you need to have an existing controller subscribing to the **target_frame** topic, reading a **geometry_msgs/PoseStamped** message.

## Running the planning    
#### 0. Source terminals
Remember to always source both the ros2_ws workspaces:
```
source $HOME/ros2_ws/install/setup.bash
```
#### 2. Run the nodes:
Assuming that a controller is already running you can run the two nodes in separate terminal:
```
ros2 run test1 eye_tracking
```
```
ros2 run test1 ur3_injection
```
#### 3. (Optional)Rviz simulation:
If your controller open an Rviz window you can visualize the eye adding a Marker.
- Add a Marker
- In the topic write **visualization_marker**
