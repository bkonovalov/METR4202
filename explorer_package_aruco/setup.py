import os
from glob import glob

from setuptools import setup

package_name = 'metr4202_explore'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@uq.edu.au',
    description='Frontier exploration + ArUco search for TurtleBot3 (METR4202)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_detector = metr4202_explore.frontier_detector:main',
            'waypoint_commander = metr4202_explore.waypoint_commander:main',
            'aruco_detector = metr4202_explore.aruco_detector:main',
        ],
    },
)
