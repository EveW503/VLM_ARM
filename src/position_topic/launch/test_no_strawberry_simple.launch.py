import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    controller_package_share_dir = get_package_share_directory("lerobot_controller")
    moveit_package_share_dir = get_package_share_directory("lerobot_moveit")
    gazebo_package_share_dir = get_package_share_directory("lerobot_description")

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_share_dir, 'launch', 'so101_controller.launch.py')
        )
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_package_share_dir, 'launch', 'so101_moveit.launch.py')
        )
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_package_share_dir, 'launch', 'so101_nostrawberry.launch.py')
        )
    )

    # ── 简化版节点 (无阶段二/无观察位姿) ──

    vlm_bridge_simple = Node(
        package='position_topic',
        executable='vlm_bridge_simple',
        name='vlm_bridge_simple',
        output='screen',
        parameters=[{
            'generic_mode': True,   # 非草莓场景, 通用抓取 prompt
        }]
    )

    grab_action_simple = Node(
        package='position_topic',
        executable='grab_action_simple',
        name='grab_action_simple',
        output='screen',
        parameters=[{
            'placement_position': [0.2, -0.15, 0.25],
            'pre_grasp_z_offset': 0.08,
            'velocity_scaling': 0.3,
            'jaw_clearance': 0.0,   # Z已修正到物体中心, 此为夹爪几何微调
            'target_object_dims': [0.05, 0.05, 0.05],
        }]
    )

    return LaunchDescription([
        gazebo_launch,
        controller_launch,
        moveit_launch,
        vlm_bridge_simple,
        grab_action_simple,
    ])
