import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    gazebo_pkg = get_package_share_directory("lerobot_description")
    ctrl_pkg = get_package_share_directory("lerobot_controller")
    moveit_pkg = get_package_share_directory("lerobot_moveit")

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, "launch", "so101_gazebo.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(ctrl_pkg, "launch", "so101_controller.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, "launch", "so101_moveit.launch.py")
        )),
        Node(
            package="position_topic",
            executable="grab_action",
            name="grab_action",
            output="screen",
        ),
        Node(
            package="position_topic",
            executable="camera_axis_calibrator",
            name="camera_axis_calibrator",
            output="screen",
        ),
    ])
