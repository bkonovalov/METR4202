#!/usr/bin/env python3
"""
Frontier exploration with a persistent waypoint graph.

Behaviour
---------
1.  Subscribes to the SLAM occupancy grid (/map) and finds *frontiers*:
    free cells that touch unknown space.
2.  Groups the frontier cells into connected clusters and picks one at
    random (A -> B is chosen stochastically rather than greedily, so the
    robot does not always hug the nearest opening).
3.  Publishes the chosen cluster's representative cell as a PoseStamped
    on /goal_pose and lets Nav2 drive there.
4.  Watches the Nav2 behaviour-tree log.  When NavigateRecovery returns
    to IDLE, the goal has finished (successfully or not), so the node
    records the outcome and picks the next frontier.
5.  Every arrival point is inserted into a WaypointGraph: nodes are
    places the robot actually reached (merged if they are close to an
    existing node), edges are traversals plus spatial neighbours.  The
    graph is published as RViz markers and dumped to JSON on shutdown.

Topics
------
sub  /map              nav_msgs/OccupancyGrid            (latched)
sub  /behavior_tree_log nav2_msgs/BehaviorTreeLog
sub  /amcl_pose        geometry_msgs/PoseWithCovarianceStamped
pub  /goal_pose        geometry_msgs/PoseStamped
pub  /exploration_graph visualization_msgs/MarkerArray
pub  /frontier_markers  visualization_msgs/MarkerArray
"""

import json
import math
import random
from collections import deque
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


# --------------------------------------------------------------------------
# Waypoint graph
# --------------------------------------------------------------------------
class WaypointGraph:
    """
    Undirected graph of places the robot has occupied.

    Nodes are (x, y) in the map frame.  A new arrival within
    ``merge_radius`` of an existing node is treated as a revisit of that
    node instead of creating a duplicate, which is what turns the record
    into a graph rather than a plain path.
    """

    def __init__(self, merge_radius=0.5, edge_radius=1.5):
        self.merge_radius = merge_radius
        self.edge_radius = edge_radius
        self.nodes = {}          # id -> dict(x, y, visits, first_seen, last_seen)
        self.edges = {}          # frozenset({a, b}) -> dict(weight, kind, count)
        self._next_id = 0

    # -- nodes ------------------------------------------------------------
    def add_or_get_node(self, x, y, stamp=None):
        """Return (node_id, created) for the position (x, y)."""
        stamp = stamp if stamp is not None else datetime.utcnow().isoformat()
        nid = self.nearest_node(x, y, self.merge_radius)
        if nid is not None:
            n = self.nodes[nid]
            # running mean keeps the node centred on repeated visits
            k = n['visits']
            n['x'] = (n['x'] * k + x) / (k + 1)
            n['y'] = (n['y'] * k + y) / (k + 1)
            n['visits'] = k + 1
            n['last_seen'] = stamp
            return nid, False

        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = {
            'x': float(x),
            'y': float(y),
            'visits': 1,
            'first_seen': stamp,
            'last_seen': stamp,
        }
        return nid, True

    def nearest_node(self, x, y, max_dist=float('inf')):
        best, best_d = None, max_dist
        for nid, n in self.nodes.items():
            d = math.hypot(n['x'] - x, n['y'] - y)
            if d <= best_d:
                best, best_d = nid, d
        return best

    # -- edges ------------------------------------------------------------
    def add_edge(self, a, b, kind='traversal'):
        if a is None or b is None or a == b:
            return
        key = frozenset((a, b))
        na, nb = self.nodes[a], self.nodes[b]
        w = math.hypot(na['x'] - nb['x'], na['y'] - nb['y'])
        if key in self.edges:
            self.edges[key]['count'] += 1
            self.edges[key]['weight'] = w
            if kind == 'traversal':
                self.edges[key]['kind'] = 'traversal'
        else:
            self.edges[key] = {'weight': w, 'kind': kind, 'count': 1}

    def link_spatial_neighbours(self, nid):
        """Add proximity edges so the graph reflects local connectivity."""
        n = self.nodes[nid]
        for other, m in self.nodes.items():
            if other == nid:
                continue
            d = math.hypot(n['x'] - m['x'], n['y'] - m['y'])
            if d <= self.edge_radius:
                self.add_edge(nid, other, kind='proximity')

    # -- export -----------------------------------------------------------
    def neighbours(self, nid):
        out = []
        for key, e in self.edges.items():
            if nid in key:
                other = next(iter(key - {nid}))
                out.append((other, e['weight']))
        return out

    def to_dict(self):
        return {
            'nodes': [
                {'id': nid, **data} for nid, data in sorted(self.nodes.items())
            ],
            'edges': [
                {
                    'source': min(key),
                    'target': max(key),
                    'weight': round(e['weight'], 4),
                    'kind': e['kind'],
                    'count': e['count'],
                }
                for key, e in self.edges.items()
            ],
        }

    def save_json(self, path):
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def save_graphml(self, path):
        """Minimal GraphML so the result loads straight into networkx/gephi."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="x" for="node" attr.name="x" attr.type="double"/>',
            '  <key id="y" for="node" attr.name="y" attr.type="double"/>',
            '  <key id="visits" for="node" attr.name="visits" attr.type="int"/>',
            '  <key id="w" for="edge" attr.name="weight" attr.type="double"/>',
            '  <graph id="exploration" edgedefault="undirected">',
        ]
        for nid, n in sorted(self.nodes.items()):
            lines.append(f'    <node id="n{nid}">')
            lines.append(f'      <data key="x">{n["x"]:.4f}</data>')
            lines.append(f'      <data key="y">{n["y"]:.4f}</data>')
            lines.append(f'      <data key="visits">{n["visits"]}</data>')
            lines.append('    </node>')
        for i, (key, e) in enumerate(self.edges.items()):
            a, b = min(key), max(key)
            lines.append(f'    <edge id="e{i}" source="n{a}" target="n{b}">')
            lines.append(f'      <data key="w">{e["weight"]:.4f}</data>')
            lines.append('    </edge>')
        lines += ['  </graph>', '</graphml>']
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines))


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------
def dilate(mask, iterations):
    """4-connected binary dilation, numpy only (no scipy dependency)."""
    out = mask.copy()
    for _ in range(max(0, iterations)):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def find_frontier_mask(grid, free_thresh, occ_thresh, inflate_cells):
    """
    Frontier = free cell with at least one unknown 4-neighbour,
    excluding anything within `inflate_cells` of an obstacle.
    """
    unknown = grid < 0
    free = (grid >= 0) & (grid <= free_thresh)
    occupied = grid >= occ_thresh

    unknown_adj = np.zeros_like(unknown)
    unknown_adj[1:, :] |= unknown[:-1, :]
    unknown_adj[:-1, :] |= unknown[1:, :]
    unknown_adj[:, 1:] |= unknown[:, :-1]
    unknown_adj[:, :-1] |= unknown[:, 1:]

    blocked = dilate(occupied, inflate_cells)
    return free & unknown_adj & ~blocked


def cluster_cells(mask, min_size):
    """8-connected connected components over a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    clusters = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        queue = deque([(sy, sx)])
        cells = []
        while queue:
            cy, cx = queue.popleft()
            cells.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(cells) >= min_size:
            clusters.append(cells)
    return clusters


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------
class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        # ---- parameters -------------------------------------------------
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('goal_topic', 'goal_pose')
        self.declare_parameter('bt_log_topic', 'behavior_tree_log')
        self.declare_parameter('pose_topic', 'amcl_pose')
        self.declare_parameter('map_frame', 'map')

        self.declare_parameter('free_threshold', 20)       # <= this is free
        self.declare_parameter('occupied_threshold', 65)   # >= this is wall
        self.declare_parameter('robot_radius', 0.22)       # metres
        self.declare_parameter('min_frontier_cells', 12)
        self.declare_parameter('min_goal_distance', 0.6)   # ignore frontiers underfoot
        self.declare_parameter('max_goal_distance', 0.0)   # 0 = unlimited
        self.declare_parameter('selection_mode', 'random') # random | weighted | nearest
        self.declare_parameter('blacklist_radius', 0.5)
        self.declare_parameter('arrival_tolerance', 0.6)
        self.declare_parameter('goal_timeout', 90.0)
        self.declare_parameter('settle_time', 2.0)         # let the map update
        self.declare_parameter('node_merge_radius', 0.6)
        self.declare_parameter('edge_radius', 1.5)
        self.declare_parameter('graph_path', '/tmp/exploration_graph')
        self.declare_parameter('random_seed', -1)

        g = self.get_parameter
        self.map_frame = g('map_frame').value
        self.free_thresh = int(g('free_threshold').value)
        self.occ_thresh = int(g('occupied_threshold').value)
        self.robot_radius = float(g('robot_radius').value)
        self.min_cells = int(g('min_frontier_cells').value)
        self.min_goal_dist = float(g('min_goal_distance').value)
        self.max_goal_dist = float(g('max_goal_distance').value)
        self.selection_mode = str(g('selection_mode').value)
        self.blacklist_radius = float(g('blacklist_radius').value)
        self.arrival_tol = float(g('arrival_tolerance').value)
        self.goal_timeout = float(g('goal_timeout').value)
        self.settle_time = float(g('settle_time').value)
        self.graph_path = str(g('graph_path').value)

        seed = int(g('random_seed').value)
        self.rng = random.Random(None if seed < 0 else seed)

        # ---- state ------------------------------------------------------
        self.map_msg = None
        self.robot_xy = None
        self.busy = False               # a goal is in flight
        self.current_goal = None        # (x, y)
        self.goal_sent_time = None
        self.blacklist = []             # [(x, y), ...] unreachable frontiers
        self.last_node_id = None
        self.goals_sent = 0
        self.goals_reached = 0
        self.finished = False
        self._settle_timer = None

        self.graph = WaypointGraph(
            merge_radius=float(g('node_merge_radius').value),
            edge_radius=float(g('edge_radius').value),
        )

        # ---- interfaces -------------------------------------------------
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            OccupancyGrid, g('map_topic').value, self.map_callback, latched)
        self.create_subscription(
            BehaviorTreeLog, g('bt_log_topic').value, self.bt_log_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, g('pose_topic').value,
            self.pose_callback, latched)

        self.goal_pub = self.create_publisher(
            PoseStamped, g('goal_topic').value, 10)
        self.graph_pub = self.create_publisher(
            MarkerArray, 'exploration_graph', latched)
        self.frontier_pub = self.create_publisher(
            MarkerArray, 'frontier_markers', 10)

        # kick things off / watchdog
        self.create_timer(1.0, self.tick)

        self.get_logger().info(
            f'Frontier explorer started (selection={self.selection_mode}).')

    # ---------------------------------------------------------------- subs
    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def bt_log_callback(self, msg: BehaviorTreeLog):
        """NavigateRecovery -> IDLE means the current goal finished."""
        if not self.busy:
            return
        for event in msg.event_log:
            if (event.node_name == 'NavigateRecovery'
                    and event.current_status == 'IDLE'):
                self.on_goal_finished()
                return

    # ------------------------------------------------------------- control
    def tick(self):
        """Periodic driver: start exploring, and time out stuck goals."""
        if self.finished:
            return

        if self.busy:
            if self.goal_sent_time is not None:
                elapsed = (self.get_clock().now() - self.goal_sent_time).nanoseconds / 1e9
                if elapsed > self.goal_timeout:
                    self.get_logger().warn('Goal timed out; blacklisting it.')
                    if self.current_goal:
                        self.blacklist.append(self.current_goal)
                    self.busy = False
                    self.explore_step()
            return

        if self.map_msg is None or self.robot_xy is None:
            self.get_logger().info(
                'Waiting for map and pose...', throttle_duration_sec=5.0)
            return

        if self.goals_sent == 0:
            # seed the graph with the starting pose
            nid, _ = self.graph.add_or_get_node(*self.robot_xy)
            self.last_node_id = nid
            self.explore_step()

    def on_goal_finished(self):
        """Record the outcome, update the graph, then choose the next goal."""
        self.busy = False
        arrived = self.robot_xy

        if arrived is None or self.current_goal is None:
            self.schedule_next()
            return

        err = math.hypot(arrived[0] - self.current_goal[0],
                         arrived[1] - self.current_goal[1])

        if err <= self.arrival_tol:
            self.goals_reached += 1
            self.get_logger().info(f'Reached goal (error {err:.2f} m).')
        else:
            self.get_logger().warn(
                f'Goal not reached (off by {err:.2f} m); blacklisting.')
            self.blacklist.append(self.current_goal)

        # A node is where the robot actually ended up, not where it aimed.
        nid, created = self.graph.add_or_get_node(*arrived)
        self.graph.add_edge(self.last_node_id, nid, kind='traversal')
        self.graph.link_spatial_neighbours(nid)
        self.last_node_id = nid
        self.get_logger().info(
            f'Graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges '
            f'({"new" if created else "revisited"} node {nid}).')
        self.publish_graph_markers()

        self.schedule_next()

    def schedule_next(self):
        """Wait a moment so SLAM can fold in the new observations."""
        if self._settle_timer is not None:
            self._settle_timer.cancel()
        self._settle_timer = self.create_timer(self.settle_time, self._settle_cb)

    def _settle_cb(self):
        if self._settle_timer is not None:
            self._settle_timer.cancel()
            self._settle_timer = None
        self.explore_step()

    # ----------------------------------------------------------- frontiers
    def explore_step(self):
        if self.finished or self.busy:
            return
        if self.map_msg is None or self.robot_xy is None:
            return

        candidates = self.find_frontier_goals()
        self.publish_frontier_markers(candidates)

        if not candidates:
            self.get_logger().info(
                f'No reachable frontiers left. Exploration complete after '
                f'{self.goals_sent} goals ({self.goals_reached} reached).')
            self.finished = True
            self.save_graph()
            return

        goal = self.pick_goal(candidates)
        self.send_goal(*goal)

    def find_frontier_goals(self):
        """Return [(x, y, size, distance), ...] in the map frame."""
        msg = self.map_msg
        info = msg.info
        res = info.resolution
        grid = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)

        inflate = max(1, int(round(self.robot_radius / res)))
        mask = find_frontier_mask(grid, self.free_thresh, self.occ_thresh, inflate)
        clusters = cluster_cells(mask, self.min_cells)

        rx, ry = self.robot_xy
        out = []
        for cells in clusters:
            arr = np.array(cells, dtype=float)          # (n, 2) as (row, col)
            cy, cx = arr[:, 0].mean(), arr[:, 1].mean()

            # snap the centroid onto a real frontier cell of this cluster
            d2 = (arr[:, 0] - cy) ** 2 + (arr[:, 1] - cx) ** 2
            row, col = cells[int(np.argmin(d2))]

            x = info.origin.position.x + (col + 0.5) * res
            y = info.origin.position.y + (row + 0.5) * res
            dist = math.hypot(x - rx, y - ry)

            if dist < self.min_goal_dist:
                continue
            if self.max_goal_dist > 0.0 and dist > self.max_goal_dist:
                continue
            if any(math.hypot(x - bx, y - by) < self.blacklist_radius
                   for bx, by in self.blacklist):
                continue

            out.append((x, y, len(cells), dist))
        return out

    def pick_goal(self, candidates):
        """A -> B selection.  Random by default, so coverage is not greedy."""
        if self.selection_mode == 'nearest':
            best = min(candidates, key=lambda c: c[3])
        elif self.selection_mode == 'weighted':
            # favour big frontiers that are close, but keep it stochastic
            weights = [c[2] / max(c[3], 0.5) for c in candidates]
            best = self.rng.choices(candidates, weights=weights, k=1)[0]
        else:
            best = self.rng.choice(candidates)

        self.get_logger().info(
            f'{len(candidates)} frontier(s); chose one at '
            f'({best[0]:.2f}, {best[1]:.2f}), {best[2]} cells, {best[3]:.2f} m away.')
        return best[0], best[1]

    def send_goal(self, x, y):
        rx, ry = self.robot_xy
        yaw = math.atan2(y - ry, x - rx)
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)

        self.current_goal = (float(x), float(y))
        self.goal_sent_time = self.get_clock().now()
        self.goals_sent += 1
        self.busy = True
        self.get_logger().info(f'Goal {self.goals_sent} sent -> ({x:.2f}, {y:.2f}).')

    # ------------------------------------------------------------ markers
    def publish_graph_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        nodes = Marker()
        nodes.header.frame_id = self.map_frame
        nodes.header.stamp = stamp
        nodes.ns = 'waypoint_nodes'
        nodes.id = 0
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD
        nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.18
        nodes.pose.orientation.w = 1.0
        nodes.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.9)
        for n in self.graph.nodes.values():
            nodes.points.append(Point(x=n['x'], y=n['y'], z=0.05))
        arr.markers.append(nodes)

        edges = Marker()
        edges.header.frame_id = self.map_frame
        edges.header.stamp = stamp
        edges.ns = 'waypoint_edges'
        edges.id = 1
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.scale.x = 0.04
        edges.pose.orientation.w = 1.0
        edges.color = ColorRGBA(r=0.9, g=0.9, b=0.1, a=0.7)
        for key in self.graph.edges:
            a, b = min(key), max(key)
            na, nb = self.graph.nodes[a], self.graph.nodes[b]
            edges.points.append(Point(x=na['x'], y=na['y'], z=0.05))
            edges.points.append(Point(x=nb['x'], y=nb['y'], z=0.05))
        arr.markers.append(edges)

        self.graph_pub.publish(arr)

    def publish_frontier_markers(self, candidates):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'frontiers'
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.pose.orientation.w = 1.0
        m.color = ColorRGBA(r=0.9, g=0.2, b=0.9, a=0.8)
        for x, y, _size, _dist in candidates:
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
        arr.markers.append(m)
        self.frontier_pub.publish(arr)

    # --------------------------------------------------------------- save
    def save_graph(self):
        try:
            self.graph.save_json(self.graph_path + '.json')
            self.graph.save_graphml(self.graph_path + '.graphml')
            self.get_logger().info(
                f'Graph written to {self.graph_path}.json / .graphml '
                f'({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges).')
        except OSError as exc:
            self.get_logger().error(f'Could not save graph: {exc}')


def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    finally:
        explorer.save_graph()
        explorer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()