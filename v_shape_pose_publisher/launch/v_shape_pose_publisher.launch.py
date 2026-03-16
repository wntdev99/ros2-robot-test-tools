from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v_shape_pose_publisher',
            executable='v_shape_pose_publisher',
            name='v_shape_pose_publisher',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
