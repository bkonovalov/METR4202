import random
import rclpy
from rclpy.node import Node
from nav2_msgs.msg import BehaviorTreeLog
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from collections import deque


class FrontierGraph:
    """Undirected graph over frontier cells, used to cluster them
    into connected regions via BFS connected-components."""
    def __init__(self):
        self.adjacency = {}

    def add_node(self, cell):
        if cell not in self.adjacency:
            self.adjacency[cell] = set()

    def add_edge(self, cell_a, cell_b):
        self.add_node(cell_a)
        self.add_node(cell_b)
        self.adjacency[cell_a].add(cell_b)
        self.adjacency[cell_b].add(cell_a)

    def connected_components(self):
        visited = set()
        clusters = []
        for start in self.adjacency:
            if start in visited:
                continue
            cluster = []
            queue = deque([start])
            visited.add(start)
            while queue:
                cell = queue.popleft()
                cluster.append(cell)
                for neighbour in self.adjacency[cell]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
            clusters.append(cluster)
        return clusters


class FrontierExplorer(Node):

    STATE_MOVING_TO_INITIAL = 'moving_to_initial'
    STATE_EXPLORING = 'exploring'

    def __init__(self):
        super().__init__('frontier_explorer')

        self.bt_log_sub = self.create_subscription(
            BehaviorTreeLog, 'behavior_tree_log', self.bt_log_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)

        self.state = self.STATE_MOVING_TO_INITIAL
        self.latest_map = None
        self.visited_frontiers = set()
        self.current_robot_pose = (0.0, 0.0)  # updated from map origin as a stand-in;
                                               # swap for a real /amcl_pose or /odom subscription if available

        # Heuristic weights: bigger frontier = more attractive,
        # farther frontier = less attractive
        self.w_size = 1.0
        self.w_distance = 0.5

        self.get_logger().info('Frontier Explorer node started.')
        self.send_initial_goal()

    # ---------- Phase 1: initial A -> B move ----------

    def send_initial_goal(self, x=None, y=None):
        """Send a starting goal. Random if x, y not given (placeholder
        range — set to something sane for your map)."""
        if x is None or y is None:
            x = random.uniform(-2.0, 2.0)
            y = random.uniform(-2.0, 2.0)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.publisher_.publish(goal)
        self.get_logger().info(f'Initial goal sent: x={x:.2f}, y={y:.2f}')

    # ---------- Phase 2: react to nav completion ----------

    def bt_log_callback(self, msg: BehaviorTreeLog):
        for event in msg.event_log:
            if event.node_name == 'NavigateRecovery' and event.current_status == 'IDLE':
                if self.state == self.STATE_MOVING_TO_INITIAL:
                    self.get_logger().info('Initial move complete. Switching to frontier exploration.')
                    self.state = self.STATE_EXPLORING
                    self.try_send_next_frontier()
                elif self.state == self.STATE_EXPLORING:
                    self.get_logger().info('Frontier goal finished/aborted. Picking next frontier.')
                    self.try_send_next_frontier()

    # ---------- Phase 3: frontier detection + selection ----------

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg  # just cache it; selection happens on-demand

    def try_send_next_frontier(self):
        if self.latest_map is None:
            self.get_logger().warn('No map received yet — cannot explore.')
            return

        msg = self.latest_map
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        def index(x, y):
            return y * width + x

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        # Detect frontier cells: free cells adjacent to unknown space
        frontier_cells = set()
        for y in range(height):
            for x in range(width):
                if data[index(x, y)] != 0:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny) and data[index(nx, ny)] == -1:
                        frontier_cells.add((x, y))
                        break

        if not frontier_cells:
            self.get_logger().info('No frontiers left — exploration complete.')
            return

        # Cluster with the graph (8-connectivity to avoid diagonal splits)
        graph = FrontierGraph()
        for (x, y) in frontier_cells:
            graph.add_node((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbour = (x + dx, y + dy)
                if neighbour in frontier_cells:
                    graph.add_edge((x, y), neighbour)

        clusters = [c for c in graph.connected_components() if len(c) >= 5]
        if not clusters:
            self.get_logger().info('No sizeable frontiers found.')
            return

        # Score each cluster with the heuristic and pick the best
        best_score = None
        best_target = None
        rx, ry = self.current_robot_pose

        for cluster in clusters:
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            world_x = origin_x + cx * resolution
            world_y = origin_y + cy * resolution

            key = (round(world_x, 1), round(world_y, 1))
            if key in self.visited_frontiers:
                continue  # skip ones we've already targeted

            dist = ((world_x - rx) ** 2 + (world_y - ry) ** 2) ** 0.5
            score = self.w_size * len(cluster) - self.w_distance * dist

            if best_score is None or score > best_score:
                best_score = score
                best_target = (world_x, world_y, key)

        if best_target is None:
            self.get_logger().info('All detected frontiers already visited.')
            return

        world_x, world_y, key = best_target
        self.visited_frontiers.add(key)
        self.send_frontier_goal(world_x, world_y)

    def send_frontier_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.publisher_.publish(goal)
        self.current_robot_pose = (x, y)  # rough stand-in until goal reached
        self.get_logger().info(f'Sent frontier goal: x={x:.2f}, y={y:.2f} (score-based)')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()