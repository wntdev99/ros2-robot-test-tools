#!/usr/bin/env python3
"""
GT Localization Node
--------------------
Gazebo ground truth 기반으로 Nav2가 필요한 두 가지를 완전 대체:
  1. map → odom TF  (MCL이 하던 일)
  2. /mcl_pose      (테스트 노드가 소비)

MCL을 완전히 비활성화한 상태에서 실행.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


def _yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


class GtLocalizationNode(Node):
    def __init__(self):
        super().__init__('gt_localization')

        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self._mcl_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl_pose', 10)

        self.create_subscription(
            Odometry, '/ground_truth/odom', self._gt_callback, best_effort)

        self.get_logger().info('GT Localization Node 시작 (MCL 대체 모드)')

    def _gt_callback(self, msg: Odometry):
        # ── /mcl_pose 발행 (GT pose를 map frame으로 그대로 사용) ──
        mcl = PoseWithCovarianceStamped()
        mcl.header.stamp = msg.header.stamp
        mcl.header.frame_id = 'map'
        mcl.pose.pose = msg.pose.pose
        # GT는 완벽한 값이므로 covariance = 0
        mcl.pose.covariance = [0.0] * 36
        self._mcl_pub.publish(mcl)

        # ── map → odom TF 계산 및 broadcast ──
        try:
            t = self._tf_buffer.lookup_transform(
                'odom', 'base_footprint', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warning(f'TF 조회 실패: {e}', throttle_duration_sec=2.0)
            return

        # GT: T_map_base
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y
        gθ = _yaw_from_quat(msg.pose.pose.orientation)

        # T_odom_base
        ox = t.transform.translation.x
        oy = t.transform.translation.y
        oθ = _yaw_from_quat(t.transform.rotation)

        # T_map_odom = T_map_base × inv(T_odom_base)
        px = -math.cos(oθ) * ox - math.sin(oθ) * oy
        py =  math.sin(oθ) * ox - math.cos(oθ) * oy
        tx  = gx + math.cos(gθ) * px - math.sin(gθ) * py
        ty  = gy + math.sin(gθ) * px + math.cos(gθ) * py
        yaw = gθ - oθ

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id  = 'odom'
        tf_msg.transform.translation.x = tx
        tf_msg.transform.translation.y = ty
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = _quat_from_yaw(yaw)
        self._tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GtLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
