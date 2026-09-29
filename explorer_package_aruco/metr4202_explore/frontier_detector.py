#!/usr/bin/env python3
"""
frontier_detector.py - finds reachable frontier goals in the SLAM map.

sub  /map                  nav_msgs/OccupancyGrid   (slam_toolbox, latched)
tf   map -> base_footprint (robot position)
pub  /frontier_candidates  geometry_msgs/PoseArray  best goal FIRST
pub  /frontier_markers     visualization_msgs/MarkerArray  (RViz)

Every `update_period` seconds it runs frontier_core.find_frontiers on the
latest map and publishes one goal per frontier cluster, sorted by cost
(path length through the map minus a bonus for large frontiers).  Each
goal is already filtered for wall clearance and reachability, and its
orientation faces the unexplored area so the camera looks into it.

An EMPTY PoseArray means "no reachable frontier" - the commander uses
that to decide exploration is finished.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseArray, Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException

from metr4202_explore.frontier_core import find_frontiers, rank_candidates


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        dp = self.declare_parameter
        dp('map_topic', 'map')
        dp('global_frame', 'map')
        dp('robot_frame', 'base_footprint')
        dp('update_period', 2.0)          # s
        dp('free_threshold', 25)
        dp('occupied_threshold', 65)
        dp('robot_radius', 0.22)          # goal must be this far from walls
        dp('reach_clearance', 0.17)       # corridor clearance for reachability
        dp('min_frontier_size', 8)        # cells (8 * 0.05 = 0.4 m)
        dp('goal_search_radius', 0.75)    # m, back-off search around a frontier
        dp('min_goal_distance', 0.3)      # m, ignore frontiers under the robot
        dp('info_gain_weight', 0.5)       # 0 = pure nearest-frontier
        dp('publish_markers', True)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.global_frame = gp('global_frame')
        self.robot_frame = gp('robot_frame')
        self.free_thresh = int(gp('free_threshold'))
        self.occ_thresh = int(gp('occupied_threshold'))
        self.robot_radius = float(gp('robot_radius'))
        self.reach_clearance = float(gp('reach_clearance'))
        self.min_size = int(gp('min_frontier_size'))
        self.search_radius = float(gp('goal_search_radius'))
        self.min_goal_dist = float(gp('min_goal_distance'))
        self.gain_weight = float(gp('info_gain_weight'))
        self.publish_markers = bool(gp('publish_markers'))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, gp('map_topic'), self.map_cb, latched)
        self.cand_pub = self.create_publisher(PoseArray, 'frontier_candidates', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'frontier_markers', 10)

        self.map_msg = None
        self.create_timer(float(gp('update_period')), self.update)
        self.get_logger().info('Frontier detector started.')

    # ------------------------------------------------------------------
    def map_cb(self, msg):
        self.map_msg = msg

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f'No robot pose yet: {exc}',
                                   throttle_duration_sec=5.0)
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ------------------------------------------------------------------
    def update(self):
        msg = self.map_msg
        if msg is None:
            self.get_logger().info('Waiting for /map ...', throttle_duration_sec=5.0)
            return
        robot = self.robot_xy()
        if robot is None:
            return

        info = msg.info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        grid = np.asarray(msg.data, dtype=np.int8).astype(np.int16)
        grid = grid.reshape(info.height, info.width)

        rx, ry = robot
        r_row, r_col = int((ry - oy) / res), int((rx - ox) / res)
        if not (0 <= r_row < info.height and 0 <= r_col < info.width):
            self.get_logger().warn('Robot is outside the map grid.',
                                   throttle_duration_sec=5.0)
            return

        t0 = time.monotonic()
        cands, stats = find_frontiers(
            grid, res, (r_row, r_col),
            free_thresh=self.free_thresh, occ_thresh=self.occ_thresh,
            robot_radius=self.robot_radius, reach_clearance=self.reach_clearance,
            min_cluster_cells=self.min_size, goal_search_radius=self.search_radius)
        ranked = rank_candidates(cands, res, self.gain_weight, self.min_goal_dist)
        dt = time.monotonic() - t0

        to_xy = lambda rc: (ox + (rc[1] + 0.5) * res, oy + (rc[0] + 0.5) * res)  # noqa: E731

        out = PoseArray()
        out.header.frame_id = self.global_frame
        out.header.stamp = self.get_clock().now().to_msg()
        goals_xy = []
        for c in ranked:
            gx, gy = to_xy(c['goal'])
            ax, ay = to_xy(c['anchor'])
            if c['goal'] != c['anchor']:
                yaw = math.atan2(ay - gy, ax - gx)     # face the frontier
            else:
                yaw = math.atan2(gy - ry, gx - rx)     # face travel direction
            p = Pose()
            p.position.x, p.position.y = float(gx), float(gy)
            (p.orientation.x, p.orientation.y,
             p.orientation.z, p.orientation.w) = yaw_to_quat(yaw)
            out.poses.append(p)
            goals_xy.append((gx, gy))
        self.cand_pub.publish(out)

        self.get_logger().info(
            f'{len(ranked)} reachable frontier goal(s) '
            f'[{stats["clusters"]} clusters, {stats["unreachable_clusters"]} unreachable, '
            f'{stats["frontier_cells"]} frontier cells] in {dt * 1000:.0f} ms',
            throttle_duration_sec=4.0)

        if self.publish_markers:
            self.publish_marker_array(goals_xy)

    # ------------------------------------------------------------------
    def publish_marker_array(self, goals_xy):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = stamp
        m.ns, m.id = 'frontier_goals', 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
        m.points = [Point(x=x, y=y, z=0.05) for x, y in goals_xy]
        arr.markers.append(m)

        best = Marker()
        best.header.frame_id = self.global_frame
        best.header.stamp = stamp
        best.ns, best.id = 'frontier_best', 1
        if goals_xy:
            best.type = Marker.SPHERE
            best.action = Marker.ADD
            best.pose.position.x, best.pose.position.y = goals_xy[0]
            best.pose.position.z = 0.1
            best.pose.orientation.w = 1.0
            best.scale.x = best.scale.y = best.scale.z = 0.3
            best.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            best.action = Marker.DELETE
        arr.markers.append(best)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
