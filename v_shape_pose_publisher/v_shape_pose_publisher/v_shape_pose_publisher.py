import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class VShapePosePublisher(Node):
    def __init__(self):
        super().__init__('v_shape_pose_publisher')

        self.publisher_ = self.create_publisher(PoseStamped, '/v_shape_pose', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('V-shape pose publisher started (sim_time)')

    def timer_callback(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = 3.0057975121233698
        msg.pose.position.y = 0.00803454730519568
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VShapePosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
