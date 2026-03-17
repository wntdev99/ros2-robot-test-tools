from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('n_trials', default_value='5'),
        Node(
            package='docking_accuracy_test',
            executable='docking_accuracy_test_node',
            output='screen',
            parameters=[{'n_trials': LaunchConfiguration('n_trials')}],
        ),
    ])
