import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

class WaypointCommander(BasicNavigator):
    def __init__(self):
        super().__init__()
        # have goal is a boolean to check if the robot is currently moving to a goal
        self.have_goal = False
        self.goal_start_time = None
        self.goal_timeout = 15.0
        self.blacklist = []
        self.blacklist_radius = 0.5
        self.current_goal = None

        self.subscription = self.create_subscription(
            PoseArray,
            "frontier_candidates",
            self.goal_callback,
            10
        )

        self.create_timer(0.5, self.check_progress)
    
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

    def goal_callback(self, msg):
        # check if robot is moving towards a goal
        if not self.have_goal:
            for pose in msg.poses:
                x, y = pose.position.x, pose.position.y

                if any(self.distance((x,y), point) < self.blacklist_radius for point in self.blacklist):
                    continue

                # robot is not moving towards a goal
                # send the robot a goal 
                self.goToPose(self.create_pose(x, y))
                self.current_goal = (x, y)
                self.have_goal = True
                self.goal_start_time = self.get_clock().now()
                return

    def check_progress(self):
        # check if robot is moving to a goal and whetther the goals has been reached
        if self.have_goal and self.isTaskComplete():
            result = self.getResult()
            if result == TaskResult.FAILED:
                self.blacklist.append(self.current_goal)
            self.have_goal = False
            self.current_goal = None

        elif self.have_goal:
            # check if the robot has been moving towards a goal for too long
            elapsed_time = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
            if elapsed_time > self.goal_timeout:
                self.blacklist.append(self.current_goal)
                self.get_logger().warn("Goal timeout reached. Cancelling goal.")
                self.cancelTask()
                self.have_goal = False

def main():
    rclpy.init()
    waypoint_commander = WaypointCommander()
    rclpy.spin(waypoint_commander)
    waypoint_commander.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()