import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('\n Pose received: [ position: (x:%f, y:%f, z:%f), orientation (x:%f, y:%f, z:%f, w:%f)]\n'
                                % (msg.pose.position.x,
                                    msg.pose.position.y,
                                    msg.pose.position.z,
                                    msg.pose.orientation.x,
                                    msg.pose.orientation.y,
                                    msg.pose.orientation.z,
                                    msg.pose.orientation.w))                                    


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()