# metr4202_explore

Frontier-based exploration + ArUco target search for a TurtleBot3 Waffle Pi
running slam_toolbox and Nav2 (default parameters).

```
 slam_toolbox ──/map──► frontier_detector ──/frontier_candidates──► waypoint_commander ──NavigateToPose/Spin──► Nav2
                              ▲ TF map→base_footprint                         │
                                                                              └──/exploration_complete──┐
 camera ──/camera/image_raw──► aruco_detector ──► /aruco_markers (RViz) + /tmp/aruco_markers.json ◄─────┘
```

| Node | Job |
|---|---|
| `frontier_detector` | Finds frontier clusters, keeps only goals that are clear of walls **and** connected to the robot through known free space, ranks them by path length through the map, publishes best-first. |
| `waypoint_commander` | State machine around the Nav2 `navigate_to_pose` action: sends goals, detects failure (rejected / aborted / timeout / no progress), blacklists bad goals, cancels goals whose frontier was explored en route, spins 360° on arrival, returns home when done. |
| `aruco_detector` | Detects 6x6 ArUco markers, estimates their pose with `solvePnP`, transforms them into `map`, takes the median of repeated sightings. |

## Build

```bash
cd ~/ros2_ws/src
unzip metr4202_explore.zip          # or copy the folder here
cd ~/ros2_ws
colcon build --symlink-install --packages-select metr4202_explore
source install/setup.bash
```

## Run (simulation, typical Humble setup — use your course's bringup if different)

```bash
# T1: world
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
# T2: Nav2 (no AMCL - SLAM provides map->odom)
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True
# T3: SLAM
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True
# T4: RViz (add MarkerArray displays for /frontier_markers and /aruco_markers)
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz
# T5: this package
ros2 launch metr4202_explore explore.launch.py
```

Real robot: `ros2 launch metr4202_explore explore.launch.py use_sim_time:=false`.

Nodes can also be run one at a time while debugging:

```bash
ros2 run metr4202_explore frontier_detector --ros-args -p use_sim_time:=true
ros2 topic echo /frontier_candidates --once
```

## Check before the first run

* Camera topics: `ros2 topic list | grep camera` — change `image_topic` /
  `camera_info_topic` in `config/explore_params.yaml` if they differ.
* Camera frame: `ros2 topic echo /camera/image_raw --field header.frame_id --once`.
  It should be an *optical* frame (z forward). If it is e.g. `camera_rgb_frame`
  (x forward), set `camera_frame: camera_rgb_optical_frame` if that exists in
  TF, otherwise `frame_is_optical: false`.
* Robot frame: `robot_frame` defaults to `base_footprint` (TurtleBot3).

## Test without ROS

```bash
python3 test/test_frontier_core.py      # unit tests
python3 tools/sim_explore.py            # random maze, fake lidar, prints coverage, saves PNG
python3 tools/sim_explore.py --size 10 --seed 4 --weight 0
```

## Tuning

| Symptom | Try |
|---|---|
| Robot ignores narrow corridors | lower `reach_clearance` (0.12–0.15) |
| Nav2 rejects goals near walls | raise `robot_radius` to 0.25 |
| Too much back-and-forth | `info_gain_weight: 0.0` (pure nearest frontier) |
| Chases tiny noise gaps | raise `min_frontier_size` |
| Slow overall | `spin_on_arrival: false` (but the camera sees less) |
| Markers localised poorly | lower `max_range`, raise `min_detections` |
