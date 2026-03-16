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
V_SHAPE_TIP = (3.006, 0.008)   # V자 안쪽 꼭지점

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


def _entrance_angle_data(tf_abs):
    """V자 입구 선-footprint 앞부분 각도 계산에 필요한 모든 데이터를 반환.

    Returns:
        dict(cx, cy, sp1, sp2, angle_deg, theta_start, diff) 또는 None
    """
    n = len(tf_abs)
    if n < 3:
        return None

    lx, ly = V_SHAPE_LEFT
    rx_, ry_ = V_SHAPE_RIGHT

    def dist_pt_seg(px, py, ax_, ay_, bx, by):
        abx, aby = bx - ax_, by - ay_
        apx, apy = px - ax_, py - ay_
        denom = abx ** 2 + aby ** 2
        if denom < 1e-12:
            return math.sqrt(apx ** 2 + apy ** 2)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        return math.sqrt((px - ax_ - t * abx) ** 2 + (py - ay_ - t * aby) ** 2)

    dists = [dist_pt_seg(p[0], p[1], lx, ly, rx_, ry_) for p in tf_abs]
    ci = dists.index(min(dists))
    cx, cy = tf_abs[ci]

    ev_x, ev_y = lx - rx_, ly - ry_
    ev_len = math.sqrt(ev_x ** 2 + ev_y ** 2)
    ev_ux, ev_uy = ev_x / ev_len, ev_y / ev_len

    edges = [
        (tf_abs[(ci - 1) % n], tf_abs[ci]),
        (tf_abs[ci], tf_abs[(ci + 1) % n]),
    ]

    def parallel_cos(e):
        dx = e[1][0] - e[0][0]
        dy = e[1][1] - e[0][1]
        elen = math.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        return abs(ev_ux * dx / elen + ev_uy * dy / elen)

    front_edge = max(edges, key=parallel_cos)
    fp1, fp2 = front_edge
    fv_x = fp2[0] - fp1[0]
    fv_y = fp2[1] - fp1[1]
    fv_len = math.sqrt(fv_x ** 2 + fv_y ** 2 + 1e-12)
    fv_ux, fv_uy = fv_x / fv_len, fv_y / fv_len

    half = fv_len / 2
    sp1 = (cx - ev_ux * half, cy - ev_uy * half)
    sp2 = (cx + ev_ux * half, cy + ev_uy * half)

    dot = max(-1.0, min(1.0, ev_ux * fv_ux + ev_uy * fv_uy))
    angle_deg = math.degrees(math.acos(abs(dot)))

    if dot < 0:
        fv_ux, fv_uy = -fv_ux, -fv_uy

    theta_start = math.atan2(ev_uy, ev_ux)
    theta_end = math.atan2(fv_uy, fv_ux)
    diff = wrap_to_pi(theta_end - theta_start)

    return dict(cx=cx, cy=cy, sp1=sp1, sp2=sp2,
                angle_deg=angle_deg, theta_start=theta_start, diff=diff)


def compute_entrance_angle_deg(footprint_shape, gt_x, gt_y, gt_yaw) -> float:
    """footprint body frame 좌표와 GT 포즈로 입구 각도(도)를 반환. CSV 기록용."""
    tf = []
    for fx, fy in footprint_shape:
        rx, ry = rotate_point(fx, fy, gt_yaw)
        tf.append((gt_x + rx, gt_y + ry))
    data = _entrance_angle_data(tf)
    return data['angle_deg'] if data is not None else float('nan')


def _draw_entrance_angle(ax, tf_abs):
    """_entrance_angle_data 결과를 matplotlib Axes에 시각화."""
    data = _entrance_angle_data(tf_abs)
    if data is None:
        return
    cx, cy = data['cx'], data['cy']
    sp1, sp2 = data['sp1'], data['sp2']
    angle_deg = data['angle_deg']
    theta_start, diff = data['theta_start'], data['diff']

    ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]],
            color='magenta', linewidth=2.0, zorder=9)

    arc_r = 0.08
    steps = 30
    thetas = [theta_start + diff * k / steps for k in range(steps + 1)]
    ax.plot(
        [cx + arc_r * math.cos(t) for t in thetas],
        [cy + arc_r * math.sin(t) for t in thetas],
        color='magenta', linewidth=1.5, zorder=10,
    )

    mid_theta = theta_start + diff / 2
    tx = cx + arc_r * 2.4 * math.cos(mid_theta)
    ty = cy + arc_r * 2.4 * math.sin(mid_theta)
    ax.text(tx, ty, f'{angle_deg:.1f}°', fontsize=9, color='magenta',
            ha='center', va='center', zorder=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


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
        # CSV: 세션 누적 파일 (append), PNG: 세션별 파일 (타임스탬프)
        self._csv_path = os.path.join(self._result_dir, 'docking_accuracy_results.csv')
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

        # CSV 헤더 (파일 없을 때만 작성)
        self._ensure_csv_header()

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

    _CSV_FIELDS = [
        'trial', 'dock_success',
        'target_x', 'target_y', 'target_yaw',
        'gt_x', 'gt_y', 'gt_yaw',
        'mcl_x', 'mcl_y', 'mcl_yaw',
        'gt_x_error_m', 'gt_y_error_m', 'gt_yaw_error_rad',
        'mcl_x_error_m', 'mcl_y_error_m', 'mcl_yaw_error_rad',
        'entrance_angle_deg',
        'png_path',
    ]

    def _ensure_csv_header(self):
        """파일이 없을 때만 헤더를 작성 (세션 간 누적 append 방식)."""
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self._CSV_FIELDS)

    def _append_csv(self, row: dict):
        """CSV에 한 행 추가 (append)."""
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
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

            # 8. 입구 각도 계산
            with self._data_lock:
                fp_shape_snap = self._footprint_shape
            entrance_angle_deg = float('nan')
            if fp_shape_snap is not None and gt_snap is not None:
                entrance_angle_deg = compute_entrance_angle_deg(
                    fp_shape_snap,
                    gt_snap.pose.pose.position.x,
                    gt_snap.pose.pose.position.y,
                    quaternion_to_yaw(gt_snap.pose.pose.orientation),
                )

            # 9. 오차 계산 및 CSV 기록
            row = self._compute_and_record(
                trial_idx + 1, dock_success, gt_snap, mcl_snap,
                dock_pose_snap, entrance_angle_deg,
            )
            self._results.append(row)

            # 10. RViz2 MarkerArray 발행
            self._publish_markers()

            # 결과 출력
            self.get_logger().info(
                f'[Trial {trial_idx + 1}] 결과: success={dock_success}, '
                f'GT_x={row["gt_x_error_m"]*100:.2f}cm  '
                f'GT_y={row["gt_y_error_m"]*100:.2f}cm  '
                f'GT_yaw={math.degrees(row["gt_yaw_error_rad"]):.2f}deg  '
                f'entrance={row["entrance_angle_deg"]:.2f}deg'
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
        entrance_angle_deg: float,
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

        # x/y 오차 (부호 있음: target - actual)
        def axis_err(t, r):
            return nan if (math.isnan(t) or math.isnan(r)) else (t - r)

        def yaw_err(tyaw, ryaw):
            return nan if (math.isnan(tyaw) or math.isnan(ryaw)) else wrap_to_pi(tyaw - ryaw)

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
            'gt_x_error_m':    axis_err(target_x, gt_x),
            'gt_y_error_m':    axis_err(target_y, gt_y),
            'gt_yaw_error_rad': yaw_err(target_yaw, gt_yaw),
            'mcl_x_error_m':   axis_err(target_x, mcl_x),
            'mcl_y_error_m':   axis_err(target_y, mcl_y),
            'mcl_yaw_error_rad': yaw_err(target_yaw, mcl_yaw),
            'entrance_angle_deg': entrance_angle_deg,
            'png_path': self._png_path,
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

            # 오차에 따른 색상 (xy 유클리드 거리로 계산)
            gx_e = row['gt_x_error_m']
            gy_e = row['gt_y_error_m']
            xy_err = math.sqrt(gx_e**2 + gy_e**2) if not (math.isnan(gx_e) or math.isnan(gy_e)) else float('nan')
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
        def vld(key):
            return [r[key] for r in self._results if not math.isnan(r[key])]

        def mean(lst):
            return sum(lst) / len(lst) if lst else float('nan')

        self.get_logger().info(
            f'\n{"="*50}\n'
            f'=== 테스트 완료: {len(self._results)}회 ===\n'
            f'GT  평균 x 오차:   {mean(vld("gt_x_error_m"))*100:.2f} cm\n'
            f'GT  평균 y 오차:   {mean(vld("gt_y_error_m"))*100:.2f} cm\n'
            f'GT  평균 yaw 오차: {math.degrees(mean([abs(v) for v in vld("gt_yaw_error_rad")])):.2f} deg\n'
            f'MCL 평균 x 오차:   {mean(vld("mcl_x_error_m"))*100:.2f} cm\n'
            f'MCL 평균 y 오차:   {mean(vld("mcl_y_error_m"))*100:.2f} cm\n'
            f'MCL 평균 yaw 오차: {math.degrees(mean([abs(v) for v in vld("mcl_yaw_error_rad")])):.2f} deg\n'
            f'평균 입구 각도:    {mean(vld("entrance_angle_deg")):.2f} deg\n'
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

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], hspace=0.4)
        fig.suptitle('Docking Accuracy Test Results', fontsize=14)

        # ── subplot 1: Top-down 2D (좌측 전체) ──────────────────
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.set_title('Top-down View (map frame)')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # V자 전체 형상: 좌측 팔, 우측 팔, 입구 일자 구조물(점선)
        ax1.plot(
            [V_SHAPE_TIP[0], V_SHAPE_LEFT[0]],
            [V_SHAPE_TIP[1], V_SHAPE_LEFT[1]],
            color='gold', linewidth=3, solid_capstyle='round', label='V-shape', zorder=5,
        )
        ax1.plot(
            [V_SHAPE_TIP[0], V_SHAPE_RIGHT[0]],
            [V_SHAPE_TIP[1], V_SHAPE_RIGHT[1]],
            color='gold', linewidth=3, solid_capstyle='round', zorder=5,
        )
        ax1.plot(
            [V_SHAPE_LEFT[0], V_SHAPE_RIGHT[0]],
            [V_SHAPE_LEFT[1], V_SHAPE_RIGHT[1]],
            color='gold', linewidth=2, linestyle='--', zorder=5,
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

        colors = plt.cm.tab10.colors

        with self._data_lock:
            footprint_shape = self._footprint_shape
            gt_paths = [list(p) for p in self._gt_path_points]

        entrance_ref_labeled = False

        for i, row in enumerate(self._results):
            color = colors[i % len(colors)]

            # GT 경로 (도킹 중 궤적, 점선)
            if i < len(gt_paths) and len(gt_paths[i]) >= 2:
                xs = [p[0] for p in gt_paths[i]]
                ys = [p[1] for p in gt_paths[i]]
                ax1.plot(xs, ys, '--', color=color, linewidth=1.5, alpha=0.85,
                         label=f'Trial {row["trial"]} path', zorder=6)
                ax1.plot(xs[0], ys[0], 'o', color=color, markersize=5, zorder=7)

            if math.isnan(row['gt_x']):
                continue

            # footprint polygon
            tf = None
            if footprint_shape is not None:
                tf = []
                for fx, fy in footprint_shape:
                    rx, ry = rotate_point(fx, fy, row['gt_yaw'])
                    tf.append((row['gt_x'] + rx, row['gt_y'] + ry))
                poly = MplPolygon(tf, closed=True, facecolor=color,
                                  edgecolor=color, alpha=0.4, zorder=7)
                ax1.add_patch(poly)
            else:
                ax1.plot(row['gt_x'], row['gt_y'], 'o', color=color,
                         markersize=8, alpha=0.8, zorder=7)

            # yaw 방향 벡터 (화살표)
            arrow_len = 0.28
            adx = arrow_len * math.cos(row['gt_yaw'])
            ady = arrow_len * math.sin(row['gt_yaw'])
            ax1.annotate(
                '', xy=(row['gt_x'] + adx, row['gt_y'] + ady),
                xytext=(row['gt_x'], row['gt_y']),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.0, mutation_scale=15),
                zorder=12,
            )

            # V자 입구 선 → 최근접 꼭지점에 평행이동 + 앞부분 라인과 각도 표시
            if tf is not None:
                if not entrance_ref_labeled:
                    # legend용 더미 선 (magenta)
                    ax1.plot([], [], color='magenta', linewidth=2,
                             label='Entrance ref + angle')
                    entrance_ref_labeled = True
                _draw_entrance_angle(ax1, tf)

        ax1.legend(loc='upper left', fontsize=7)

        # ── subplot 2: GT x/y 오차 막대 그래프 (우측 상단) ──────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('GT Position Error per Trial')

        trial_nums = [r['trial'] for r in self._results]
        gt_x_cm = [r['gt_x_error_m'] * 100 for r in self._results]
        gt_y_cm = [r['gt_y_error_m'] * 100 for r in self._results]

        x = np.arange(len(trial_nums))
        w = 0.3  # 2개 bar

        ax2.bar(x - 0.5*w, gt_x_cm, w, label='GT x [cm]', color='#e74c3c', alpha=0.85)
        ax2.bar(x + 0.5*w, gt_y_cm, w, label='GT y [cm]', color='#c0392b', alpha=0.85)

        def valid_mean(vals):
            v = [val for val in vals if not math.isnan(val)]
            return sum(v) / len(v) if v else None

        for vals, color, lbl in [
            (gt_x_cm, '#e74c3c', 'GT x mean'),
            (gt_y_cm, '#c0392b', 'GT y mean'),
        ]:
            m = valid_mean(vals)
            if m is not None:
                ax2.axhline(m, color=color, linestyle='--', linewidth=1.2,
                            alpha=0.7, label=f'{lbl} ({m:.1f} cm)')

        ax2.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Error [cm] (signed)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'T{n}' for n in trial_nums])
        ax2.legend(fontsize=7)
        ax2.grid(True, axis='y', alpha=0.3)

        # ── subplot 3: GT yaw 오차 막대 그래프 (우측 하단) ───────
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title('GT Yaw Error per Trial')

        gt_yaw_deg = [r['gt_yaw_error_rad'] * 180 / math.pi for r in self._results]

        ax3.bar(x, gt_yaw_deg, 0.5, label='GT yaw [deg]', color='#e67e22', alpha=0.85)

        m_yaw = valid_mean(gt_yaw_deg)
        if m_yaw is not None:
            ax3.axhline(m_yaw, color='#e67e22', linestyle='--', linewidth=1.2,
                        alpha=0.7, label=f'GT yaw mean ({m_yaw:.2f} deg)')

        ax3.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('Error [deg] (signed)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{n}' for n in trial_nums])
        ax3.legend(fontsize=7)
        ax3.grid(True, axis='y', alpha=0.3)

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
