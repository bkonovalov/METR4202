import rclpy
from rclpy.node import Node

from nav2_msgs.msg import BehaviorTreeLog
from geometry_msgs.msg import PoseStamped


class WaypointCycler(Node):

    def __init__(self):
        super().__init__('waypoint_cycler')

        # Subscribe to the Nav2 behaviour-tree status log
        self.subscription = self.create_subscription(
            BehaviorTreeLog,
            'behavior_tree_log',
            self.bt_log_callback,
            10
        )
        self.subscription  # prevent unused-variable warning

        # Publish waypoint goals
        self.publisher_ = self.create_publisher(
            PoseStamped,
            'goal_pose',
            10
        )

        # Keep track of how many waypoints have been sent
        self.waypoint_counter = 0

        # Waypoint 0
        p0 = PoseStamped()
        p0.header.frame_id = 'map'
        p0.pose.position.x = 1.7
        p0.pose.position.y = -0.5
        p0.pose.orientation.w = 1.0

        # Waypoint 1
        p1 = PoseStamped()
        p1.header.frame_id = 'map'
        p1.pose.position.x = -0.6
        p1.pose.position.y = 1.8
        p1.pose.orientation.w = 1.0

        self.waypoints = [p0, p1]

        self.get_logger().info('Waypoint Cycler node started.')


    def bt_log_callback(self, msg: BehaviorTreeLog):
        """
        Called whenever a BehaviorTreeLog message is received.

        When NavigateRecovery becomes IDLE, Nav2 has either completed
        the current goal or stopped after a navigation failure.
        """
        for event in msg.event_log:
            if (
                event.node_name == 'NavigateRecovery'
                and event.current_status == 'IDLE'
            ):
                self.get_logger().info(
                    'Navigation finished/aborted. Sending next waypoint.'
                )
                self.send_waypoint()


    def send_waypoint(self):
        """Publish the next waypoint, alternating between p0 and p1."""

        self.waypoint_counter += 1

        # Match the cycling logic used in the practical sheet
        if self.waypoint_counter % 2:
            waypoint_index = 1
        else:
            waypoint_index = 0

        waypoint = self.waypoints[waypoint_index]

        # Update timestamp before publishing
        waypoint.header.stamp = self.get_clock().now().to_msg()

        self.publisher_.publish(waypoint)

        self.get_logger().info(
            f'Sent waypoint {waypoint_index}: '
            f'x={waypoint.pose.position.x:.2f}, '
            f'y={waypoint.pose.position.y:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)

    waypoint_cmder = WaypointCycler()

    try:
        rclpy.spin(waypoint_cmder)
    except KeyboardInterrupt:
        pass
    finally:
        waypoint_cmder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
