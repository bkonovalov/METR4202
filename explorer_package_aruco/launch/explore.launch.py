import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('metr4202_explore')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params = LaunchConfiguration('params_file')

    common = [params, {'use_sim_time': use_sim_time}]
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg, 'config', 'explore_params.yaml')),
        Node(package='metr4202_explore', executable='frontier_detector',
             name='frontier_detector', parameters=common, output='screen'),
        Node(package='metr4202_explore', executable='waypoint_commander',
             name='waypoint_commander', parameters=common, output='screen'),
        Node(package='metr4202_explore', executable='aruco_detector',
             name='aruco_detector', parameters=common, output='screen'),
    ])
