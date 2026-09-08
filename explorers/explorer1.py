import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from collections import deque


class FrontierGraph:
    """
    Simple undirected graph mapping (x, y) grid cells -> set of
    neighbouring (x, y) frontier cells. Used to cluster frontier
    cells into connected regions.
    """
    def __init__(self):
        self.adjacency = {}  # dict[(x, y)] -> set[(x, y)]

    def add_node(self, cell):
        if cell not in self.adjacency:
            self.adjacency[cell] = set()

    def add_edge(self, cell_a, cell_b):
        self.add_node(cell_a)
        self.add_node(cell_b)
        self.adjacency[cell_a].add(cell_b)
        self.adjacency[cell_b].add(cell_a)

    def connected_components(self):
        """Return a list of clusters, each a list of (x, y) cells."""
        visited = set()
        clusters = []

        for start in self.adjacency:
            if start in visited:
                continue
            # BFS out from this cell to grab its whole cluster
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
    def __init__(self):
        super().__init__('frontier_explorer')

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            PoseStamped,
            'goal_pose',
            10
        )

        self.visited_frontiers = set()  # avoid re-sending the same target
        self.get_logger().info('Frontier Explorer node started.')

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        data = msg.data  # flat list, -1 unknown, 0 free, 100 occupied
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        def index(x, y):
            return y * width + x

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        # --- Step 1: find frontier cells ---
        # A frontier cell is FREE and has at least one UNKNOWN neighbour
        frontier_cells = set()
        for y in range(height):
            for x in range(width):
                if data[index(x, y)] != 0:
                    continue  # only care about free cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny) and data[index(nx, ny)] == -1:
                        frontier_cells.add((x, y))
                        break

        if not frontier_cells:
            self.get_logger().info('No frontiers left — exploration complete.')
            return

        # --- Step 2: build graph, connecting adjacent frontier cells ---
        graph = FrontierGraph()
        for (x, y) in frontier_cells:
            graph.add_node((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbour = (x + dx, y + dy)
                if neighbour in frontier_cells:
                    graph.add_edge((x, y), neighbour)

        # --- Step 3: cluster into distinct frontier regions ---
        clusters = graph.connected_components()

        # Discard tiny clusters (likely noise)
        clusters = [c for c in clusters if len(c) >= 5]
        if not clusters:
            self.get_logger().info('No sizeable frontiers found.')
            return

        # --- Step 4: pick a target — e.g. the largest frontier's centroid ---
        best_cluster = max(clusters, key=len)
        cx = sum(c[0] for c in best_cluster) / len(best_cluster)
        cy = sum(c[1] for c in best_cluster) / len(best_cluster)

        world_x = origin_x + cx * resolution
        world_y = origin_y + cy * resolution

        target_key = (round(world_x, 1), round(world_y, 1))
        if target_key in self.visited_frontiers:
            return  # already tried this one recently
        self.visited_frontiers.add(target_key)

        self.send_goal(world_x, world_y)

    def send_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.publisher_.publish(goal)
        self.get_logger().info(f'Sent frontier goal: x={x:.2f}, y={y:.2f}')


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