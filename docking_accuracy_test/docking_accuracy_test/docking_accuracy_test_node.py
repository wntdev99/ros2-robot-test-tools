"""
Docking Accuracy Test Node

Nav2 docking server를 사용하여 전진 도킹 정확도를 정량적으로 측정.
매 시도마다 원점 복귀 → 도킹 시도 → GT/MCL 기록의 반복으로 통계를 수집.
결과: CSV + 실시간 RViz2 MarkerArray + matplotlib PNG
"""

import csv
import math
import os
import sys
import threading
import time
from datetime import datetime

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
)
from nav_msgs.msg import Odometry
from nav2_msgs.action import DockRobot, NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PolygonStamped

# DockRobot Feedback state 상수
NAV_TO_STAGING_POSE = 1
INITIAL_PERCEPTION = 2
CONTROLLING = 3
WAIT_FOR_CHARGE = 4
RETRY = 5

# V자 구조물 좌표 (시각화용)
V_SHAPE_LEFT = (3.25, 0.428)
V_SHAPE_RIGHT = (3.25, -0.427)

# 오차 임계값 (색상 분류)
ERROR_GREEN_THRESHOLD = 0.05   # m
ERROR_YELLOW_THRESHOLD = 0.10  # m

# 경로 샘플링 간격 (초)
PATH_SAMPLE_INTERVAL = 0.1


def quaternion_to_yaw(q: Quaternion) -> float:
    """쿼터니언을 yaw(라디안)로 변환."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """yaw(라디안)를 쿼터니언으로 변환."""
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


def wrap_to_pi(angle: float) -> float:
    """각도를 [-π, π] 범위로 정규화."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotate_point(x: float, y: float, yaw: float):
    """점 (x, y)을 yaw 각도로 회전."""
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return (cos_y * x - sin_y * y, sin_y * x + cos_y * y)


class DockingAccuracyTestNode(Node):
    """도킹 정확도 측정 노드."""

    def __init__(self):
        super().__init__('docking_accuracy_test_node')

        # 파라미터
        self.declare_parameter('n_trials', 5)
        self._n_trials = self.get_parameter('n_trials').value

        # 최신 센서 데이터
        self._latest_gt = None           # Odometry (GT)
        self._latest_mcl = None          # PoseWithCovarianceStamped (MCL)
        self._latest_dock_pose = None    # PoseStamped (dock target)
        self._footprint_shape = None     # 상대 footprint polygon [(x, y), ...]

        # 경로 버퍼 (trial별, GT만, 도킹 중에만 수집)
        self._gt_path_points = []    # [[(x, y), ...], ...]
        self._current_trial = -1
        self._last_gt_sample_time = 0.0
        self._collecting_path = False  # 도킹 action 중에만 True

        # 결과 데이터
        self._results = []  # dict 목록

        # 결과 저장 경로
        pkg_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # ament_python 설치 환경에서 패키지 루트를 src 디렉토리로 설정
        self._result_dir = os.path.join(
            '/home/jeongmin/ros2_ws/src/temp_package/docking_accuracy_test/result'
        )
        os.makedirs(self._result_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(
            self._result_dir, f'docking_accuracy_{timestamp}.csv'
        )
        self._png_path = os.path.join(
            self._result_dir, f'docking_accuracy_{timestamp}.png'
        )

        # QoS 설정
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # 구독자
        self._gt_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self._gt_callback,
            sensor_qos,
        )
        self._mcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/mcl_pose',
            self._mcl_callback,
            sensor_qos,
        )
        self._dock_pose_sub = self.create_subscription(
            PoseStamped,
            '/dock_pose',
            self._dock_pose_callback,
            latched_qos,
        )
        self._footprint_sub = self.create_subscription(
            PolygonStamped,
            '/local_costmap/published_footprint',
            self._footprint_callback,
            sensor_qos,
        )

        # 발행자
        self._marker_pub = self.create_publisher(
            MarkerArray,
            '/docking_accuracy_markers',
            10,
        )

        # Action 클라이언트
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._dock_client = ActionClient(self, DockRobot, '/dock_robot')

        # 잠금
        self._data_lock = threading.Lock()
        self._input_event = threading.Event()
        self._input_quit = False

        # CSV 초기화
        self._init_csv()

        self.get_logger().info(
            f'DockingAccuracyTestNode 초기화 완료. n_trials={self._n_trials}'
        )
        self.get_logger().info(f'CSV: {self._csv_path}')
        self.get_logger().info(f'PNG: {self._png_path}')

        # 테스트 루프 시작 (별도 스레드)
        self._test_thread = threading.Thread(target=self._test_loop, daemon=True)
        self._test_thread.start()

    # ─── 구독 콜백 ───────────────────────────────────────────────

    def _gt_callback(self, msg: Odometry):
        with self._data_lock:
            self._latest_gt = msg
            self._maybe_sample_path_gt(msg)

    def _mcl_callback(self, msg: PoseWithCovarianceStamped):
        with self._data_lock:
            self._latest_mcl = msg

    def _dock_pose_callback(self, msg: PoseStamped):
        with self._data_lock:
            self._latest_dock_pose = msg
        self.get_logger().debug(
            f'/dock_pose 수신: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f})'
        )

    def _footprint_callback(self, msg: PolygonStamped):
        with self._data_lock:
            if self._footprint_shape is not None:
                return  # one-shot latch
            if self._latest_gt is None:
                return  # GT 없으면 상대 변환 불가, 다음 콜백에서 재시도
            robot_x = self._latest_gt.pose.pose.position.x
            robot_y = self._latest_gt.pose.pose.position.y
            robot_yaw = quaternion_to_yaw(self._latest_gt.pose.pose.orientation)
            pts = []
            for pt in msg.polygon.points:
                # 절대 좌표 → 로봇 body frame 상대 좌표로 변환
                dx = pt.x - robot_x
                dy = pt.y - robot_y
                rx, ry = rotate_point(dx, dy, -robot_yaw)
                pts.append((rx, ry))
            self._footprint_shape = pts
        self.get_logger().info(
            f'footprint 캡처 완료: {len(self._footprint_shape)}개 꼭지점'
        )

    # ─── 경로 샘플링 ─────────────────────────────────────────────

    def _maybe_sample_path_gt(self, msg: Odometry):
        """GT 경로 포인트 샘플링 (도킹 중에만, 10Hz, lock 내부에서 호출)."""
        if not self._collecting_path:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_gt_sample_time < PATH_SAMPLE_INTERVAL:
            return
        if self._current_trial < 0 or self._current_trial >= len(self._gt_path_points):
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._gt_path_points[self._current_trial].append((x, y))
        self._last_gt_sample_time = now

    # ─── CSV ─────────────────────────────────────────────────────

    def _init_csv(self):
        """CSV 파일 헤더 작성."""
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial', 'dock_success',
                'target_x', 'target_y', 'target_yaw',
                'gt_x', 'gt_y', 'gt_yaw',
                'mcl_x', 'mcl_y', 'mcl_yaw',
                'gt_xy_error_m', 'gt_yaw_error_rad',
                'mcl_xy_error_m', 'mcl_yaw_error_rad',
            ])

    def _append_csv(self, row: dict):
        """CSV에 한 행 추가."""
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'trial', 'dock_success',
                'target_x', 'target_y', 'target_yaw',
                'gt_x', 'gt_y', 'gt_yaw',
                'mcl_x', 'mcl_y', 'mcl_yaw',
                'gt_xy_error_m', 'gt_yaw_error_rad',
                'mcl_xy_error_m', 'mcl_yaw_error_rad',
            ])
            writer.writerow(row)

    # ─── 테스트 루프 ─────────────────────────────────────────────

    def _test_loop(self):
        """메인 테스트 루프 (별도 스레드)."""
        # Action 서버 대기
        self.get_logger().info('Action 서버 대기 중...')
        self._nav_client.wait_for_server()
        self._dock_client.wait_for_server()
        self.get_logger().info('Action 서버 연결 완료.')

        # 최초 센서 데이터 대기
        self.get_logger().info('센서 데이터 대기 중...')
        timeout = 30.0
        start = time.time()
        while time.time() - start < timeout:
            with self._data_lock:
                gt_ok = self._latest_gt is not None
                mcl_ok = self._latest_mcl is not None
            if gt_ok and mcl_ok:
                break
            time.sleep(0.5)
        else:
            self.get_logger().error('센서 데이터 수신 타임아웃. 종료합니다.')
            rclpy.shutdown()
            return

        self.get_logger().info('테스트 시작!')

        # 사용자 입력 스레드 시작
        input_thread = threading.Thread(
            target=self._input_loop, daemon=True
        )
        input_thread.start()

        for trial_idx in range(self._n_trials):
            self.get_logger().info(
                f'\n{"="*50}\n[Trial {trial_idx + 1}/{self._n_trials}] 시작\n{"="*50}'
            )

            # 1. trial 경로 버퍼 초기화
            with self._data_lock:
                self._current_trial = trial_idx
                self._gt_path_points.append([])

            # 2. 원점 복귀
            self.get_logger().info('원점으로 이동 중...')
            nav_success = self._navigate_to_origin()
            if not nav_success:
                self.get_logger().warn('원점 이동 실패. 계속 진행합니다.')

            # 3. dock_pose 초기화, 경로 수집 시작
            with self._data_lock:
                self._latest_dock_pose = None
                self._collecting_path = True

            # 4. DockRobot action 전송
            self.get_logger().info('도킹 시도 중...')
            dock_success = self._dock_robot()

            # 도킹 완료 후 경로 수집 중지
            with self._data_lock:
                self._collecting_path = False

            # 5, 6, 7. 스냅샷 수집
            with self._data_lock:
                gt_snap = self._latest_gt
                mcl_snap = self._latest_mcl
                dock_pose_snap = self._latest_dock_pose

            # 8. 오차 계산 및 CSV 기록
            row = self._compute_and_record(
                trial_idx + 1, dock_success, gt_snap, mcl_snap, dock_pose_snap
            )
            self._results.append(row)

            # 9. RViz2 MarkerArray 발행
            self._publish_markers()

            # 결과 출력
            self.get_logger().info(
                f'[Trial {trial_idx + 1}] 결과: success={dock_success}, '
                f'GT_xy={row["gt_xy_error_m"]:.4f}m, '
                f'GT_yaw={row["gt_yaw_error_rad"]:.4f}rad, '
                f'MCL_xy={row["mcl_xy_error_m"]:.4f}m'
            )

            # 10. 사용자 입력 대기
            if trial_idx < self._n_trials - 1:
                self.get_logger().info(
                    "Enter: 다음 테스트 | 'q'+Enter: 종료"
                )
                self._input_event.clear()
                self._input_event.wait()

                if self._input_quit:
                    self.get_logger().info('사용자 요청으로 테스트 중단.')
                    break

        # 완료 처리
        self._finalize()

    def _wait_future(self, future, timeout_sec: float = 300.0) -> bool:
        """MultiThreadedExecutor가 이미 spin 중인 환경에서 future 완료 대기.

        rclpy.spin_until_future_complete()는 별도 executor를 생성해 spin을 시도하므로
        메인 executor와 충돌(교착 상태)이 발생한다. 대신 폴링 방식으로 대기한다.
        """
        start = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error(f'future 대기 타임아웃 ({timeout_sec}s)')
                return False
            time.sleep(0.05)
        return future.done()

    def _navigate_to_origin(self) -> bool:
        """NavigateToPose로 원점 (0, 0, yaw=0)으로 이동."""
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = 0.0
        goal.pose.pose.position.y = 0.0
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation = yaw_to_quaternion(0.0)

        future = self._nav_client.send_goal_async(goal)
        if not self._wait_future(future):
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('NavigateToPose goal 거부됨.')
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future):
            return False

        result = result_future.result()
        if result is None:
            return False

        from action_msgs.msg import GoalStatus
        return result.status == GoalStatus.STATUS_SUCCEEDED

    def _dock_robot(self) -> bool:
        """DockRobot action을 전송하고 완료까지 대기."""
        goal = DockRobot.Goal()
        goal.use_dock_id = True
        goal.dock_id = 'charging_dock'
        goal.navigate_to_staging_pose = True
        goal.max_staging_time = 1000.0

        send_future = self._dock_client.send_goal_async(
            goal,
            feedback_callback=self._dock_feedback_callback,
        )
        if not self._wait_future(send_future):
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('DockRobot goal 거부됨.')
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout_sec=600.0):
            return False

        result = result_future.result()
        if result is None:
            return False

        from action_msgs.msg import GoalStatus
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if success:
            self.get_logger().info('도킹 성공.')
        else:
            self.get_logger().warn(f'도킹 실패. status={result.status}')
        return success

    def _dock_feedback_callback(self, feedback_msg):
        """DockRobot feedback 콜백."""
        state = feedback_msg.feedback.state
        if state == INITIAL_PERCEPTION:
            self.get_logger().info('스테이징 포즈 도착, 도크 인식 시작.')
        elif state == CONTROLLING:
            self.get_logger().info('도킹 제어 중...')
        elif state == WAIT_FOR_CHARGE:
            self.get_logger().info('충전 대기 중...')
        elif state == RETRY:
            self.get_logger().warn('도킹 재시도 중...')

    # ─── 오차 계산 ───────────────────────────────────────────────

    def _compute_and_record(
        self,
        trial: int,
        dock_success: bool,
        gt_snap: Odometry,
        mcl_snap: PoseWithCovarianceStamped,
        dock_pose_snap: PoseStamped,
    ) -> dict:
        """오차 계산 후 CSV 기록 및 dict 반환."""
        nan = float('nan')

        # GT 포즈
        if gt_snap is not None:
            gt_x = gt_snap.pose.pose.position.x
            gt_y = gt_snap.pose.pose.position.y
            gt_yaw = quaternion_to_yaw(gt_snap.pose.pose.orientation)
        else:
            gt_x = gt_y = gt_yaw = nan

        # MCL 포즈
        if mcl_snap is not None:
            mcl_x = mcl_snap.pose.pose.position.x
            mcl_y = mcl_snap.pose.pose.position.y
            mcl_yaw = quaternion_to_yaw(mcl_snap.pose.pose.orientation)
        else:
            mcl_x = mcl_y = mcl_yaw = nan

        # 목표 도킹 포즈
        if dock_pose_snap is not None:
            target_x = dock_pose_snap.pose.position.x
            target_y = dock_pose_snap.pose.position.y
            target_yaw = quaternion_to_yaw(dock_pose_snap.pose.orientation)
        else:
            target_x = target_y = target_yaw = nan
            self.get_logger().warn(
                '/dock_pose 캡처 실패: target 포즈를 NaN으로 기록합니다.'
            )

        # 오차 계산
        def xy_error(tx, ty, rx, ry):
            if any(math.isnan(v) for v in [tx, ty, rx, ry]):
                return nan
            return math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)

        def yaw_err(tyaw, ryaw):
            if math.isnan(tyaw) or math.isnan(ryaw):
                return nan
            return wrap_to_pi(tyaw - ryaw)

        gt_xy_error = xy_error(target_x, target_y, gt_x, gt_y)
        gt_yaw_error = yaw_err(target_yaw, gt_yaw)
        mcl_xy_error = xy_error(target_x, target_y, mcl_x, mcl_y)
        mcl_yaw_error = yaw_err(target_yaw, mcl_yaw)

        row = {
            'trial': trial,
            'dock_success': dock_success,
            'target_x': target_x,
            'target_y': target_y,
            'target_yaw': target_yaw,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'gt_yaw': gt_yaw,
            'mcl_x': mcl_x,
            'mcl_y': mcl_y,
            'mcl_yaw': mcl_yaw,
            'gt_xy_error_m': gt_xy_error,
            'gt_yaw_error_rad': gt_yaw_error,
            'mcl_xy_error_m': mcl_xy_error,
            'mcl_yaw_error_rad': mcl_yaw_error,
        }
        self._append_csv(row)
        return row

    # ─── RViz2 MarkerArray ────────────────────────────────────────

    def _publish_markers(self):
        """누적된 모든 trial 데이터로 MarkerArray 발행."""
        marker_array = MarkerArray()
        marker_id = 0

        # 1. V자 입구 일자 구조물 (LINE_STRIP, 노랑, 고정)
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'v_shape'
        line_marker.id = marker_id
        marker_id += 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        p1 = Point()
        p1.x = V_SHAPE_LEFT[0]
        p1.y = V_SHAPE_LEFT[1]
        p1.z = 0.1
        p2 = Point()
        p2.x = V_SHAPE_RIGHT[0]
        p2.y = V_SHAPE_RIGHT[1]
        p2.z = 0.1
        line_marker.points = [p1, p2]
        marker_array.markers.append(line_marker)

        # 2. 목표 도킹 포즈 (SPHERE, 흰색)
        with self._data_lock:
            dock_pose_snap = self._latest_dock_pose

        if dock_pose_snap is not None:
            sphere_marker = Marker()
            sphere_marker.header.frame_id = 'map'
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.ns = 'dock_target'
            sphere_marker.id = marker_id
            marker_id += 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose = dock_pose_snap.pose
            sphere_marker.scale.x = 0.10
            sphere_marker.scale.y = 0.10
            sphere_marker.scale.z = 0.10
            sphere_marker.color.r = 1.0
            sphere_marker.color.g = 1.0
            sphere_marker.color.b = 1.0
            sphere_marker.color.a = 1.0
            marker_array.markers.append(sphere_marker)

        # 3. 각 trial GT 경로 (LINE_STRIP, 빨강)
        with self._data_lock:
            gt_paths = [list(p) for p in self._gt_path_points]

        for i, pts in enumerate(gt_paths):
            if len(pts) < 2:
                continue
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'gt_path'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.015
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.8
            for px, py in pts:
                p = Point()
                p.x = px
                p.y = py
                p.z = 0.05
                m.points.append(p)
            marker_array.markers.append(m)

        # 5. 각 trial GT 최종 포즈 ARROW + TEXT (오차 색상)
        for row in self._results:
            if math.isnan(row['gt_x']):
                continue

            # 오차에 따른 색상
            xy_err = row['gt_xy_error_m']
            if math.isnan(xy_err):
                r, g, b = 0.5, 0.5, 0.5
            elif xy_err < ERROR_GREEN_THRESHOLD:
                r, g, b = 0.0, 1.0, 0.0
            elif xy_err < ERROR_YELLOW_THRESHOLD:
                r, g, b = 1.0, 1.0, 0.0
            else:
                r, g, b = 1.0, 0.0, 0.0

            # ARROW
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = 'gt_final_pose'
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = row['gt_x']
            arrow.pose.position.y = row['gt_y']
            arrow.pose.position.z = 0.05
            arrow.pose.orientation = yaw_to_quaternion(row['gt_yaw'])
            arrow.scale.x = 0.30
            arrow.scale.y = 0.06
            arrow.scale.z = 0.06
            arrow.color.r = r
            arrow.color.g = g
            arrow.color.b = b
            arrow.color.a = 1.0
            marker_array.markers.append(arrow)

            # TEXT
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = 'gt_error_text'
            txt.id = marker_id
            marker_id += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = row['gt_x']
            txt.pose.position.y = row['gt_y']
            txt.pose.position.z = 0.30
            txt.scale.z = 0.10
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            err_str = (
                f'NaN' if math.isnan(xy_err) else f'{xy_err*100:.1f}cm'
            )
            txt.text = f'T{row["trial"]}: {err_str}'
            marker_array.markers.append(txt)

        self._marker_pub.publish(marker_array)

    # ─── 사용자 입력 ─────────────────────────────────────────────

    def _input_loop(self):
        """사용자 입력 대기 스레드."""
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if line is None:
                    break
                line = line.strip()
                if line.lower() == 'q':
                    self._input_quit = True
                self._input_event.set()
            except EOFError:
                break

    # ─── 최종 처리 ───────────────────────────────────────────────

    def _finalize(self):
        """테스트 완료 후 통계 출력 및 PNG 저장."""
        if not self._results:
            self.get_logger().warn('기록된 결과가 없습니다.')
            rclpy.shutdown()
            return

        # 통계 계산
        gt_xy_errors = [
            r['gt_xy_error_m'] for r in self._results
            if not math.isnan(r['gt_xy_error_m'])
        ]
        gt_yaw_errors = [
            abs(r['gt_yaw_error_rad']) for r in self._results
            if not math.isnan(r['gt_yaw_error_rad'])
        ]
        mcl_xy_errors = [
            r['mcl_xy_error_m'] for r in self._results
            if not math.isnan(r['mcl_xy_error_m'])
        ]
        mcl_yaw_errors = [
            abs(r['mcl_yaw_error_rad']) for r in self._results
            if not math.isnan(r['mcl_yaw_error_rad'])
        ]

        def mean(lst):
            return sum(lst) / len(lst) if lst else float('nan')

        self.get_logger().info(
            f'\n{"="*50}\n'
            f'=== 테스트 완료: {len(self._results)}회 ===\n'
            f'GT  평균 xy 오차:  {mean(gt_xy_errors)*100:.2f} cm\n'
            f'GT  평균 yaw 오차: {math.degrees(mean(gt_yaw_errors)):.2f} deg\n'
            f'MCL 평균 xy 오차:  {mean(mcl_xy_errors)*100:.2f} cm\n'
            f'MCL 평균 yaw 오차: {math.degrees(mean(mcl_yaw_errors)):.2f} deg\n'
            f'CSV: {self._csv_path}\n'
            f'PNG: {self._png_path}\n'
            f'{"="*50}'
        )

        # matplotlib PNG 저장
        self._save_png()

        rclpy.shutdown()

    def _save_png(self):
        """matplotlib으로 결과 PNG 저장."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
            import numpy as np
        except ImportError as e:
            self.get_logger().error(f'matplotlib import 실패: {e}')
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Docking Accuracy Test Results', fontsize=14)

        # ── subplot 1: Top-down 2D ──────────────────────────────
        ax1 = axes[0]
        ax1.set_title('Top-down View (map frame)')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # V자 입구 구조물 (일자 선)
        ax1.plot(
            [V_SHAPE_LEFT[0], V_SHAPE_RIGHT[0]],
            [V_SHAPE_LEFT[1], V_SHAPE_RIGHT[1]],
            'y-', linewidth=3, label='V-shape wall', zorder=5,
        )

        # 목표 도킹 포즈 ★
        if self._results:
            tx = self._results[0]['target_x']
            ty = self._results[0]['target_y']
            if not math.isnan(tx):
                ax1.plot(
                    tx, ty, '*', color='white', markersize=15,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label='Target dock pose', zorder=10,
                )

        # 색상 팔레트 (trial별)
        colors = plt.cm.tab10.colors

        with self._data_lock:
            footprint_shape = self._footprint_shape
            gt_paths = [list(p) for p in self._gt_path_points]

        for i, row in enumerate(self._results):
            color = colors[i % len(colors)]

            # GT 경로 선
            if i < len(gt_paths) and len(gt_paths[i]) >= 2:
                xs = [p[0] for p in gt_paths[i]]
                ys = [p[1] for p in gt_paths[i]]
                ax1.plot(
                    xs, ys,
                    '-', color=color, linewidth=1.2, alpha=0.7,
                    label=f'Trial {row["trial"]} path',
                )
                # 시작점 ○
                ax1.plot(xs[0], ys[0], 'o', color=color, markersize=5, zorder=6)

            # GT 최종 footprint polygon (body frame 상대좌표 → gt_pose로 변환)
            if not math.isnan(row['gt_x']) and footprint_shape is not None:
                transformed = []
                for fx, fy in footprint_shape:
                    rx, ry = rotate_point(fx, fy, row['gt_yaw'])
                    transformed.append((row['gt_x'] + rx, row['gt_y'] + ry))
                poly = MplPolygon(
                    transformed,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.5,
                )
                ax1.add_patch(poly)
            elif not math.isnan(row['gt_x']):
                ax1.plot(
                    row['gt_x'], row['gt_y'],
                    'o', color=color, markersize=8, alpha=0.8,
                )

        ax1.legend(loc='upper left', fontsize=7)

        # ── subplot 2: 오차 막대 그래프 ─────────────────────────
        ax2 = axes[1]
        ax2.set_title('Docking Error per Trial')

        trial_nums = [r['trial'] for r in self._results]
        gt_xy = [r['gt_xy_error_m'] * 100 for r in self._results]    # cm
        mcl_xy = [r['mcl_xy_error_m'] * 100 for r in self._results]  # cm
        gt_yaw = [
            abs(r['gt_yaw_error_rad']) * 180 / math.pi
            for r in self._results
        ]
        mcl_yaw = [
            abs(r['mcl_yaw_error_rad']) * 180 / math.pi
            for r in self._results
        ]

        x = np.arange(len(trial_nums))
        width = 0.2

        ax2.bar(x - 1.5*width, gt_xy,  width, label='GT xy [cm]',   color='#e74c3c', alpha=0.8)
        ax2.bar(x - 0.5*width, mcl_xy, width, label='MCL xy [cm]',  color='#3498db', alpha=0.8)
        ax2.bar(x + 0.5*width, gt_yaw, width, label='GT yaw [deg]', color='#e67e22', alpha=0.8)
        ax2.bar(x + 1.5*width, mcl_yaw, width, label='MCL yaw [deg]', color='#9b59b6', alpha=0.8)

        # 평균선
        valid_gt_xy = [v for v in gt_xy if not math.isnan(v)]
        valid_mcl_xy = [v for v in mcl_xy if not math.isnan(v)]
        valid_gt_yaw = [v for v in gt_yaw if not math.isnan(v)]
        valid_mcl_yaw = [v for v in mcl_yaw if not math.isnan(v)]

        if valid_gt_xy:
            ax2.axhline(
                sum(valid_gt_xy) / len(valid_gt_xy),
                color='#e74c3c', linestyle='--', linewidth=1.5,
                alpha=0.6, label=f'GT xy mean ({sum(valid_gt_xy)/len(valid_gt_xy):.1f} cm)',
            )
        if valid_mcl_xy:
            ax2.axhline(
                sum(valid_mcl_xy) / len(valid_mcl_xy),
                color='#3498db', linestyle='--', linewidth=1.5,
                alpha=0.6, label=f'MCL xy mean ({sum(valid_mcl_xy)/len(valid_mcl_xy):.1f} cm)',
            )

        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Error')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'T{n}' for n in trial_nums])
        ax2.legend(fontsize=8)
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.get_logger().info(f'PNG 저장 완료: {self._png_path}')


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = DockingAccuracyTestNode()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
