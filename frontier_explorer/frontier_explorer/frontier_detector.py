import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose

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

    def create_pose(self, x, y):
        waypoint = Pose()
        waypoint.position.x = x
        waypoint.position.y = y
        waypoint.orientation.w = 1.0
        return waypoint

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

    def map_callback(self, msg):
        self.get_logger().info("recieved message")
        goals = []

        # define map parameters
        width = msg.info.width
        height = msg.info.height

        # to iterate through the map data, the cell index
        # is calculated as index = row * width + column
        for row in range(height):
            for col in range (width):
                cell_value = msg.data[row * width + col]

                if cell_value != 0:
                    continue
            
                count = self.adjacent_to_unknown(msg.data, row, col, width, height)
                if count == 0:
                    continue

                x = msg.info.origin.position.x + (col + 0.5) * msg.info.resolution
                y = msg.info.origin.position.y + (row + 0.5) * msg.info.resolution
                pose = self.create_pose(x,y)

                goals.append((pose, count))

        goals.sort(key=lambda x: x[1], reverse=True)
        self.publisher.publish(PoseArray(header=msg.header, poses=[goal[0] for goal in goals]))

def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()