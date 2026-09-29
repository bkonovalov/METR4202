import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav2_msgs.msg import BehaviorTreeLog
from tf2_ros import Buffer, TransformListener, TransformException

class WaypointCommander(Node):
    def __init__(self):
        super().__init__('waypoint_commander')
        
        # have goal is a boolean to check if the robot is currently moving to a goal
        self.have_goal = False
        self.current_goal = None
        self.goal_start_time = None
        self.goal_timeout = 15.0
        self.tolerance = 0.5
        self.blacklist = []
        self.blacklist_radius = 0.5
        self.latest_candidates = []
        self.current_goal = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        self.create_timer(0.5, self.check_progress)
    
    def candidates_callback(self, msg):
        self.latest_candidates = msg.poses

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
        
        err = self.distance(pos, self.current_goal)

        if err > self.tolerance:
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

    def check_progress(self):
        # check if robot has a goal to move towards
        if self.have_goal:
            elapsed_time = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
            if elapsed_time > self.goal_timeout:
                self.blacklist.append(self.current_goal)
                self.get_logger().warn("Goal timeout reached. Cancelling goal.")
                self.have_goal = False
            return
        
        for pose in self.latest_candidates:
            x, y = pose.position.x, pose.position.y
            if any(self.distance((x,y), point) < self.blacklist_radius for point in self.blacklist):
                continue
            self.send_goal(x, y)
            return
    
    def send_goal(self, x, y):
        pose = self.create_pose(x,y)
        self.goal_pub.publish(pose)
        self.current_goal = (x, y)
        self.have_goal = True
        self.goal_start_time = self.get_clock().now()

def main():
    rclpy.init()
    node = WaypointCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()