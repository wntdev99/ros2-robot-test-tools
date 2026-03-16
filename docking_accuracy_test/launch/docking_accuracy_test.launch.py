from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    n_trials_arg = DeclareLaunchArgument(
        'n_trials',
        default_value='5',
        description='최대 도킹 테스트 반복 횟수',
    )

    node = Node(
        package='docking_accuracy_test',
        executable='docking_accuracy_test_node',
        name='docking_accuracy_test_node',
        output='screen',
        parameters=[{
            'n_trials': LaunchConfiguration('n_trials'),
        }],
    )

    return LaunchDescription([
        n_trials_arg,
        node,
    ])
