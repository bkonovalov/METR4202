import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav2_msgs.msg import BehaviorTreeLog
from tf2_ros import Buffer, TransformListener, TransformException
from nav_msgs.msg import OccupancyGrid

class WaypointCommander(Node):
    def __init__(self):
        super().__init__('waypoint_commander')
        
        # have goal is a boolean to check if the robot is currently moving to a goal
        self.have_goal = False
        self.current_goal = None
        self.goal_start_time = None
        self.goal_timeout = 50.0
        self.tolerance = 0.5
        self.blacklist = []
        self.blacklist_radius = 0.5
        self.latest_candidates = []
        self.current_goal = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_position = None
        self.last_position_time = None
        self.stall_check_interval = 3.0
        self.stall_distance_threshold = 0.1

        self.latest_costmap = None
        self.lethal_cost_threshold = 99

        self.sub_frontier = self.create_subscription(
            PoseArray,
            "frontier_candidates",
            self.candidates_callback,
            10
        )

        self.sub_bt = self.create_subscription(
            BehaviorTreeLog,
            "behavior_tree_log",
            self.bt_log_callback,
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            "goal_pose",
            10
        )

        costmap_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)
        
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            "global_costmap/costmap",
            self.costmap_callback,
            costmap_qos
        )

        self.create_timer(0.5, self.check_progress)
    
    def candidates_callback(self, msg):
        self.latest_candidates = msg.poses

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return (transform.transform.translation.x, transform.transform.translation.y)
        except TransformException as e:
            self.get_logger().warn(f"Could not get robot position: {e}")
            return None

    def bt_log_callback(self, msg):
        if not self.have_goal:
            return

        self.check_stall_timeout()
        if not self.have_goal:
            return

        for event in msg.event_log:
            if event.node_name == "NavigateRecovery" and event.current_status == "IDLE":
                self.on_goal_finished()
                return
    
    def on_goal_finished(self):
        pos = self.get_robot_position()

        if pos is None:
            self.have_goal = False
            self.current_goal = None
            return
        
        #err = self.distance(pos, self.current_goal)

        #if err > self.tolerance:
        self.blacklist.append(self.current_goal)
        
        self.have_goal = False
        self.current_goal = None

    def distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def create_pose(self, x, y):
        waypoint = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = 'map'
        waypoint.pose.position.x = x
        waypoint.pose.position.y = y
        waypoint.pose.orientation.w = 1.0
        return waypoint

    def is_reachable(self, x, y):
        if self.latest_costmap is None:
            return True
        
        info = self.latest_costmap.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return False
        
        index = row * info.width + col
        cost = self.latest_costmap.data[index]

        return cost < self.lethal_cost_threshold

    def check_stall_timeout(self):
        elapsed_time = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
        if elapsed_time > self.goal_timeout:
            self.blacklist.append(self.current_goal)
            self.get_logger().warn("Goal timeout reached. Cancelling goal.")
            self.have_goal = False
            return

        since_last_check = (self.get_clock().now() - self.last_position_time).nanoseconds / 1e9
        if since_last_check > self.stall_check_interval:
            curr_pos = self.get_robot_position()
            old_pos = self.last_position

            self.last_position = curr_pos
            self.last_position_time = self.get_clock().now()

            if curr_pos is not None and old_pos is not None:
                dist_moved = self.distance(curr_pos, old_pos)
                if dist_moved < self.stall_distance_threshold:
                    self.blacklist.append(self.current_goal)
                    self.get_logger().warn("Robot appears to be stalled. Cancelling goal.")
                    self.have_goal = False

                    return

    def check_progress(self):
        # check if robot has a goal to move towards
        if self.have_goal:
            self.check_stall_timeout()
        else:
            for pose in self.latest_candidates:
                x, y = pose.position.x, pose.position.y
                if any(self.distance((x,y), point) < self.blacklist_radius for point in self.blacklist):
                    continue
                if not self.is_reachable(x, y):
                    continue
                self.send_goal(x, y)
                return
    
    def send_goal(self, x, y):
        pose = self.create_pose(x,y)
        self.goal_pub.publish(pose)
        self.current_goal = (x, y)
        self.have_goal = True
        self.goal_start_time = self.get_clock().now()

        # record the robot's current position and time pose was sent
        self.last_position = self.get_robot_position()
        self.last_position_time = self.get_clock().now()

def main():
    rclpy.init()
    node = WaypointCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()