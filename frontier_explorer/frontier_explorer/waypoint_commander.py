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
        self.blacklist_radius = 0.4
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

        self.commited_area_centre = None
        self.commited_area_radius = 1.5
        self.distance_weight = 0.05
        self.latest_scores = []

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

        blacklist_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        self.blacklist_pub = self.create_publisher(
            PoseArray,
            "blacklist",
            blacklist_qos
        )

        self.create_timer(0.5, self.check_progress)
    
    def candidates_callback(self, msg):
        # Update the latest candidates and their scores
        self.latest_candidates = msg.poses
        num_candidates = len(self.latest_candidates)
        self.get_logger().info(f"Received {num_candidates} frontier candidates")
        self.latest_scores = [num_candidates - i for i in range(num_candidates)]

    def costmap_callback(self, msg):
        # Update the latest costmap
        self.latest_costmap = msg

    def get_robot_position(self):
        # Get the robot's current position in the map frame
        try:
            # Lookup the transform from 'map' to 'base_link' to get the robot's position
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return (transform.transform.translation.x, transform.transform.translation.y)
        except TransformException as e:
            # Log a warning if the transform cannot be found
            self.get_logger().warn(f"Could not get robot position: {e}")
            return None

    def blacklist_publish(self):
        # Publish the current blacklist as a PoseArray message
        poses = []
        msg = PoseArray()

        # Create a Pose for each blacklisted point and add it to the PoseArray
        for x, y in self.blacklist:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.orientation.w = 1.0
            poses.append(pose)
        
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = poses
        self.blacklist_pub.publish(msg)

    def bt_log_callback(self, msg):
        # Check if the robot has a goal to move towards
        if not self.have_goal:
            # No goal, skip checking for goal completion
            return

        # Check if robot has stalled
        # If the robot has stalled, the goal will be cancelled and the current goal will be blacklisted
        self.check_stall_timeout()
        if not self.have_goal:
            # Robot stalled, goal cancelled
            return

        # Check if the goal has been reached by looking for a "NavigateRecovery" event with status "IDLE"
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
        
        # Check if the robot is within the tolerance of the goal position
        err = self.distance(pos, self.current_goal)

        # If the robot is not within the tolerance, blacklist the goal 
        if err > self.tolerance:
            self.get_logger().info(f"Goal point blacklisted: {self.current_goal}")
            self.blacklist.append(self.current_goal)
            self.blacklist_publish()

        self.have_goal = False
        self.current_goal = None

    def distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def create_pose(self, x, y):
        # Convert x,y coordinates to PoseStamped message
        waypoint = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = 'map'
        waypoint.pose.position.x = x
        waypoint.pose.position.y = y
        waypoint.pose.orientation.w = 1.0
        return waypoint

    def is_reachable(self, x, y):
        # Check if the given x,y coordinates are reachable based on the latest costmap
        if self.latest_costmap is None:
            return True
        
        # Convert the x,y coordinates to row and column indices in the costmap
        info = self.latest_costmap.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        # Check if the row and column indices are within the bounds of the costmap
        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return False
        
        # Get the cost value from the costmap data at the calculated index
        index = row * info.width + col
        cost = self.latest_costmap.data[index]

        # Check if the cost is below the lethal cost threshold to determine reachability
        return cost < self.lethal_cost_threshold

    def check_stall_timeout(self):
        # Check if the robot has stalled or if the goal has timed out
        elapsed_time = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
        if elapsed_time > self.goal_timeout:
            self.blacklist.append(self.current_goal)
            self.blacklist_publish()
            self.get_logger().warn(f"Goal timeout reached at {self.current_goal}. Cancelling goal.")
            self.have_goal = False
            return

        # Check if the robot has moved since the last position check
        since_last_check = (self.get_clock().now() - self.last_position_time).nanoseconds / 1e9
        if since_last_check > self.stall_check_interval:
            curr_pos = self.get_robot_position()
            old_pos = self.last_position

            self.last_position = curr_pos
            self.last_position_time = self.get_clock().now()

            # Check if the robot has moved less than the stall distance threshold
            if curr_pos is not None and old_pos is not None:
                dist_moved = self.distance(curr_pos, old_pos)

                # If the robot has moved less than the stall distance threshold, blacklist the current goal and cancel it
                if dist_moved < self.stall_distance_threshold:
                    self.blacklist.append(self.current_goal)
                    self.blacklist_publish()
                    self.get_logger().warn(f"Robot appears to be stalled near {self.current_goal} (moved {dist_moved: .2f}m). Cancelling goal.")
                    self.have_goal = False

                    return

    def best_candidate(self, candidates):
        # Select the best candidate from the list of candidates based on cost and proximity to the committed area center
        if self.commited_area_centre is not None:
            local_candidates = [c for c in candidates if self.distance((c[1], c[2]), self.commited_area_centre) < self.commited_area_radius]

            # If there are local candidates within the committed area radius, select the one with the minimum cost
            if local_candidates:
                return min(local_candidates, key=lambda c: c[0])[1:3]
        
        return min(candidates, key=lambda c: c[0])[1:3]

    def check_progress(self):
        robot_pos = self.get_robot_position()

        # If the robot position cannot be obtained, log a warning and skip candidate selection
        if robot_pos is None:
            self.get_logger().warn("Could not get robot position. Skipping candidate selection.")
            return
        
        # check if robot has a goal to move towards
        if self.have_goal:
            self.check_stall_timeout()
        
        else:
            # If the robot does not have a goal, select the best candidate from the latest candidates
            candidates = []

            # Iterate through candidates to find best goal based on distance, score, and reachability
            for i, pose in enumerate(self.latest_candidates):
                x, y = pose.position.x, pose.position.y

                # Check if the candidate is blacklisted 
                if any(self.distance((x,y), point) < self.blacklist_radius for point in self.blacklist):
                    continue

                # Check if the candidate is reachable based on the latest costmap
                if not self.is_reachable(x, y):
                    continue

                # Check if the candidate is too close to the robot's current position
                if robot_pos is not None and self.distance(robot_pos, (x,y)) < self.tolerance:
                    continue

                # Calculate the cost of the candidate based on distance and score, and add it to the list of candidates
                dist = self.distance(robot_pos, (x,y))
                cost = dist - self.distance_weight * self.latest_scores[i]
                candidates.append((cost, x, y))

            if not candidates:
                # If no reachable candidates are found, log the total number of candidates and blacklisted points
                self.get_logger().info(f"No reachable candidates found. Total: {len(self.latest_candidates)}, blacklisted: {len(self.blacklist)}")
                return
            
            # Select the best candidate based on the calculated costs and send it as a goal
            x, y = self.best_candidate(candidates)
            self.send_goal(x, y)

    def send_goal(self, x, y):
        # Send a goal to the robot by publishing a PoseStamped message with the specified x,y coordinates
        pose = self.create_pose(x,y)
        self.get_logger().info(f"sending goal: ({x: .2f}, {y: .2f})")
        self.goal_pub.publish(pose)
        self.current_goal = (x, y)
        self.have_goal = True
        self.goal_start_time = self.get_clock().now()

        # record the robot's current position and time pose was sent
        self.last_position = self.get_robot_position()
        self.last_position_time = self.get_clock().now()

        # Update the committed area center to the new goal position
        self.commited_area_centre = (x, y)
        self.get_logger().info(f"Commited area centre set to: {self.commited_area_centre}")
        
def main():
    rclpy.init()
    node = WaypointCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()