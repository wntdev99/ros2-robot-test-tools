from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('n_trials', default_value='5'),
        # GT 로컬라이제이션 (MCL 대체)
        Node(
            package='docking_accuracy_test',
            executable='gt_localization_node',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
        # 도킹 정확도 테스트 노드 (기존과 동일)
        Node(
            package='docking_accuracy_test',
            executable='docking_accuracy_test_node',
            output='screen',
            parameters=[{'n_trials': LaunchConfiguration('n_trials')}],
        ),
    ])
