import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, Point

from collections import deque

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class FrontierDetector(Node):
    def __init__(self):
        # create frontier detector node
        super().__init__('frontier_detector')

        # matches QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # subscriber to occupancy map
        self.subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos
        )

        # publisher to "frontier_candidates"
        # allows a waypoint commander to read frontier nodes
        self.publisher = self.create_publisher(
            PoseArray,
            'frontier_candidates',
            10
        )

        # used to publish regions (created from flood fill) to rviz
        # region_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #     depth=1,
        # )

        # self.region_pub = self.create_publisher(
        #     MarkerArray,
        #     'frontier_regions',
        #     region_qos
        # )

        # self.max_points_per_region = 100
        # self.points_per_n_cell = 20

    def create_pose(self, x, y):
        # create a Pose from a given x, y coordinate
        waypoint = Pose()

        waypoint.position.x = x
        waypoint.position.y = y
        # fix rotation to face foward
        waypoint.orientation.w = 1.0

        return waypoint

    # used to publish regions
    # Uncomment this function and region published in init to use
    # def region_publish(self, msg, regions):

    #     colours = [ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    #                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    #                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    #                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    #                ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
    #                ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)]

    #     marker_array = MarkerArray()
    #     # create a delete marker to clear previous markers
    #     delete_marker = Marker()
    #     delete_marker.action = Marker.DELETEALL
    #     marker_array.markers.append(delete_marker)

    #     for i, region in enumerate(regions):
    #         marker = Marker()
    #         marker.header.frame_id = "map"
    #         marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.ns = "frontier_regions"
    #         marker.id = i
    #         marker.type = Marker.POINTS
    #         marker.action = Marker.ADD
    #         marker.scale.x = 0.1
    #         marker.scale.y = 0.1
    #         marker.color = colours[i % len(colours)]
            
    #         # sample points from the region if it has more than max_points_per_region
    #         if len(region) > self.max_points_per_region:
    #             step = max(1, len(region) // self.points_per_n_cell)
    #             sampled_region = region[::step]
    #             sampled_region = sampled_region[:self.max_points_per_region]
    #         else:
    #             sampled_region = region
            
    #         for row, col in sampled_region:
    #             x, y = self.grid_to_world(msg, row, col)
    #             point = Point(x=x, y=y, z=0.0)
    #             marker.points.append(point)
            
    #         marker_array.markers.append(marker)
        
    #     self.region_pub.publish(marker_array)

    def adjacent_to_unknown(self, data, row, col, width, height):
        # Iterate through adjacent cells (up, down, left, right)
        # max and min functions are to limit row and col to within the bounds
        # of the map
        total_adjacent = 0
        for r in range(max(0, row - 1), min(height, row + 2)):
            for c in range(max(0, col - 1), min(width, col + 2)):
                # skip over position of known free cell
                if r == row and c == col:
                    continue
                
                # check the occupancy map for unknown cell
                index = r * width + c
                if data[index] == -1:
                    # frontier cell found
                    total_adjacent += 1
        
        return total_adjacent
    
    def adjacent_cells(self, data, row, col, width, height):
        adjacent = []
        # Iterate through adjacent cells (up, down, left, right)
        # max and min functions are to limit row and col to within the bounds of the map
        for r in range(max(0, row - 1), min(height, row + 2)):
            for c in range(max(0, col - 1), min(width, col + 2)):
                # skip over position of known free cell
                if r == row and c == col:
                    continue
                
                adjacent.append((r, c))        
        
        return adjacent

    def best_goal_in_region(self, region):
        # find the cell in the region that is closest to the centre of the region
        centre_row = sum(r for r, c in region) / len(region)
        centre_col = sum(c for r, c in region) / len(region)

        best_cell = None
        best_dist = float('inf')

        # iterate through the cells in the region to find the closest one to the centre
        for row, col in region:
            dist = (row - centre_row)**2 + (col - centre_col)**2
            if dist < best_dist:
                best_dist = dist
                best_cell = (row, col)
        return best_cell
    
    def pick_goals(self, region, k):
        # pick k goals from the region that are as far apart as possible
        # used to handle blacklisting of goals that are unreachable or cause the robot to stall
        centre_row = sum(r for r, c in region) / len(region)
        centre_col = sum(c for r, c in region) / len(region)

        first = min(region, key=lambda cell: (cell[0] - centre_row)**2 + (cell[1] - centre_col)**2)
        chosen = [first]

        while len(chosen) < k and len(chosen) < len(region):
            best_cell = None
            best_dist = -1

            for cell in region:
                if cell in chosen:
                    continue
                
                dist = min((cell[0] - c[0])**2 + (cell[1] - c[1])**2 for c in chosen)
                if dist > best_dist:
                    best_dist = dist
                    best_cell = cell
            
            chosen.append(best_cell)
        
        return chosen

    def grid_to_world(self, msg, row, col):
        # convert grid coordinates to world coordinates
        x = msg.info.origin.position.x + (col + 0.5) * msg.info.resolution
        y = msg.info.origin.position.y + (row + 0.5) * msg.info.resolution
        return x, y

    def map_callback(self, msg):
        goals = []
        frontier_cells = set()

        width = msg.info.width
        height = msg.info.height

        # search for all candidate goals where a candidate cell is adjacent to 
        # at least 1 unknown cell
        for row in range(height):
            for col in range(width):
                cell_value = msg.data[row * width + col]

                if cell_value != 0:
                    # cell not empty
                    continue
            
                count = self.adjacent_to_unknown(msg.data, row, col, width, height)
                if count == 0:
                    # no unknown adjacent cells
                    continue

                frontier_cells.add((row, col))
        self.get_logger().info(f"Num frontier cells: {len(frontier_cells)}")

        # flood fill
        visited = set()
        regions = []

        # create a list of regions
        for cell in frontier_cells:
            if cell in visited:
                continue

            region = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                row, col = queue.popleft()
                region.append([row, col])

                for next_row, next_col, in self.adjacent_cells(msg.data, row, col, width, height):
                    if (next_row, next_col) in frontier_cells and (next_row, next_col) not in visited:
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col))
            
            regions.append(region)
        
        region_sizes = sorted((len(r) for r in regions), reverse=True)
        self.get_logger().info(f"regions: {len(regions)}, sizes: {region_sizes}")

        # pick the best goal in each region and add it to the list of goals
        for region in regions:
            score = len(region)
            for row, col in region:
                x, y = self.grid_to_world(msg, row, col)
                pose = self.create_pose(x, y)
                goals.append((pose, score))
                self.get_logger().info(f"candidate: ({x: .2f}, {y: .2f}) score={score}")

        goals.sort(key=lambda x: x[1], reverse=True)
        self.publisher.publish(PoseArray(header=msg.header, poses=[goal[0] for goal in goals]))
        
        self.region_publish(msg, regions)

    # def map_callback_old(self, msg):
    #     self.get_logger().info("recieved message")
    #     goals = []

    #     # define map parameters
    #     width = msg.info.width
    #     height = msg.info.height

    #     # to iterate through the map data, the cell index
    #     # is calculated as index = row * width + column
    #     for row in range(height):
    #         for col in range (width):
    #             cell_value = msg.data[row * width + col]

    #             if cell_value != 0:
    #                 continue
            
    #             count = self.adjacent_to_unknown(msg.data, row, col, width, height)
    #             if count == 0:
    #                 continue

    #             x = msg.info.origin.position.x + (col + 0.5) * msg.info.resolution
    #             y = msg.info.origin.position.y + (row + 0.5) * msg.info.resolution
    #             pose = self.create_pose(x,y)

    #             goals.append((pose, count))

    #     goals.sort(key=lambda x: x[1], reverse=True)
    #     self.publisher.publish(PoseArray(header=msg.header, poses=[goal[0] for goal in goals]))

def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()