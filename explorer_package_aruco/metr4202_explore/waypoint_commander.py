#!/usr/bin/env python3
"""
waypoint_commander.py - sends frontier goals to Nav2 and handles failure.

sub  /frontier_candidates   geometry_msgs/PoseArray (best first)
act  navigate_to_pose       nav2_msgs/action/NavigateToPose
act  spin                   nav2_msgs/action/Spin (optional look-around)
pub  /exploration_complete  std_msgs/Bool (latched)

State machine
-------------
WAITING_FOR_NAV2 -> IDLE -> NAVIGATING -> (SPINNING) -> IDLE -> ...
                                                     +-> DONE (no frontiers left)

* Talks to Nav2 through async action clients, so nothing ever blocks the
  executor (BasicNavigator's goToPose/isTaskComplete spin the node
  internally and must not be called from inside callbacks).
* Failure detection: goal rejected, result ABORTED, hard timeout, or no
  progress (distance_remaining not shrinking) for `progress_timeout` s.
* Failed goals go on a blacklist; any new goal within `blacklist_radius`
  of a blacklisted point is skipped.  Reached goals are also remembered,
  so a frontier that never clears (e.g. out of lidar range) is not
  revisited forever.
* If the frontier the robot is driving to disappears from the map on the
  way (it got explored early), the goal is cancelled and the next one is
  chosen - saves a lot of time in a maze.
* When `empty_rounds_to_finish` consecutive candidate messages contain no
  usable goal, failed goals are retried once, then exploration ends and
  (optionally) the robot drives back to the start (map origin).
"""
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from std_msgs.msg import Bool

WAITING, IDLE, NAVIGATING, SPINNING, DONE = (
    'WAITING_FOR_NAV2', 'IDLE', 'NAVIGATING', 'SPINNING', 'DONE')


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WaypointCommander(Node):

    def __init__(self):
        super().__init__('waypoint_commander')

        dp = self.declare_parameter
        dp('global_frame', 'map')
        dp('startup_delay', 3.0)            # s after Nav2 appears
        dp('settle_time', 1.0)              # s to let SLAM catch up after a goal
        dp('goal_timeout', 120.0)           # s hard limit per goal
        dp('progress_timeout', 30.0)        # s without getting closer -> fail
        dp('progress_epsilon', 0.10)        # m that counts as progress
        dp('blacklist_radius', 0.5)         # m
        dp('preempt_explored_goals', True)
        dp('preempt_radius', 0.75)          # m
        dp('empty_rounds_to_finish', 3)
        dp('retry_failed_once', True)
        dp('spin_on_arrival', True)         # 360 deg look-around (camera!)
        dp('spin_angle', 6.28)              # rad
        dp('spin_timeout', 25.0)            # s
        dp('return_home', True)
        dp('home_x', 0.0)
        dp('home_y', 0.0)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame = gp('global_frame')
        self.startup_delay = float(gp('startup_delay'))
        self.settle_time = float(gp('settle_time'))
        self.goal_timeout = float(gp('goal_timeout'))
        self.progress_timeout = float(gp('progress_timeout'))
        self.progress_eps = float(gp('progress_epsilon'))
        self.blacklist_radius = float(gp('blacklist_radius'))
        self.preempt = bool(gp('preempt_explored_goals'))
        self.preempt_radius = float(gp('preempt_radius'))
        self.empty_rounds_to_finish = int(gp('empty_rounds_to_finish'))
        self.retry_failed_once = bool(gp('retry_failed_once'))
        self.spin_on_arrival = bool(gp('spin_on_arrival'))
        self.spin_angle = float(gp('spin_angle'))
        self.spin_timeout = float(gp('spin_timeout'))
        self.return_home = bool(gp('return_home'))
        self.home = (float(gp('home_x')), float(gp('home_y')))

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.create_subscription(PoseArray, 'frontier_candidates', self.candidates_cb, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.done_pub = self.create_publisher(Bool, 'exploration_complete', latched)

        # ---- state ----
        self.state = WAITING
        self.nav_seen_time = None
        self.idle_since = None
        self.seq = 0                      # identifies the active action goal
        self.goal_handle = None
        self.goal_kind = None             # 'frontier' | 'home'
        self.current = None               # (x, y)
        self.goal_sent_time = None
        self.cancel_reason = None
        self.cancel_time = None
        self.ref_remaining = None
        self.last_progress_time = None
        self.failed = []                  # unreachable / timed-out goals
        self.visited = []                 # reached or already-explored goals
        self.retried = False
        self.empty_rounds = 0
        self.home_attempted = False
        self.n_sent = self.n_reached = self.n_failed = 0

        self.create_timer(0.5, self.watchdog)
        self.get_logger().info('Waypoint commander started, waiting for Nav2...')

    # ================================================================ utils
    def now(self):
        return self.get_clock().now()

    def secs_since(self, t):
        return (self.now() - t).nanoseconds / 1e9

    def is_blacklisted(self, x, y):
        return any(math.hypot(x - bx, y - by) < self.blacklist_radius
                   for bx, by in self.failed + self.visited)

    def go_idle(self, delay=0.0):
        self.state = IDLE
        self.idle_since = self.now() + Duration(seconds=delay)
        self.goal_handle = None
        self.cancel_reason = None
        self.cancel_time = None

    # ======================================================= candidate input
    def candidates_cb(self, msg):
        cands = [(p.position.x, p.position.y, yaw_from_quat(p.orientation))
                 for p in msg.poses]
        if self.state == NAVIGATING:
            self.maybe_preempt(cands)
        elif self.state == IDLE:
            if self.secs_since(self.idle_since) >= self.settle_time:
                self.dispatch(cands)

    def dispatch(self, cands):
        for x, y, yaw in cands:
            if not self.is_blacklisted(x, y):
                self.empty_rounds = 0
                self.send_nav_goal(x, y, yaw, kind='frontier')
                return

        self.empty_rounds += 1
        self.get_logger().info(
            f'No usable frontier ({len(cands)} candidates, all blacklisted) '
            f'[{self.empty_rounds}/{self.empty_rounds_to_finish}]')
        if self.empty_rounds < self.empty_rounds_to_finish:
            return
        if self.retry_failed_once and self.failed and not self.retried:
            self.get_logger().warn(
                f'Retrying {len(self.failed)} previously failed goal(s) once.')
            self.failed.clear()
            self.retried = True
            self.empty_rounds = 0
            return
        self.finish()

    def maybe_preempt(self, cands):
        if (not self.preempt or self.goal_kind != 'frontier'
                or self.cancel_reason is not None or self.goal_handle is None):
            return
        if not cands or self.secs_since(self.goal_sent_time) < 3.0:
            return                     # empty list may just be a detector glitch
        cx, cy = self.current
        if any(math.hypot(x - cx, y - cy) < self.preempt_radius for x, y, _ in cands):
            return
        self.get_logger().info('Target frontier already explored - moving on.')
        self.visited.append(self.current)
        self.cancel_current('explored', failed=False)

    # ========================================================= navigation
    def make_pose(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.header.stamp = self.now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        return ps

    def send_nav_goal(self, x, y, yaw, kind):
        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)

        self.seq += 1
        seq = self.seq
        self.state = NAVIGATING
        self.goal_kind = kind
        self.current = (x, y)
        self.goal_handle = None
        self.cancel_reason = None
        self.cancel_time = None
        self.goal_sent_time = self.now()
        self.last_progress_time = self.now()
        self.ref_remaining = None
        self.n_sent += 1
        self.get_logger().info(f'[{kind} #{self.n_sent}] -> ({x:.2f}, {y:.2f})')

        fut = self.nav_client.send_goal_async(
            goal, feedback_callback=lambda fb, s=seq: self.nav_feedback_cb(s, fb))
        fut.add_done_callback(lambda f, s=seq: self.nav_response_cb(s, f))

    def nav_response_cb(self, seq, future):
        if seq != self.seq:
            return
        handle = future.result()
        if not handle.accepted:
            # usually Nav2 not fully active yet - back off, don't blacklist
            self.get_logger().warn('Goal rejected by Nav2, retrying shortly.')
            self.go_idle(delay=2.0)
            return
        self.goal_handle = handle
        if self.cancel_reason is not None:        # cancel requested meanwhile
            handle.cancel_goal_async()
        handle.get_result_async().add_done_callback(
            lambda f, s=seq: self.nav_result_cb(s, f))

    def nav_feedback_cb(self, seq, fb_msg):
        if seq != self.seq:
            return
        d = fb_msg.feedback.distance_remaining
        if d <= 0.0:
            return                                 # no plan yet
        if self.ref_remaining is None or d < self.ref_remaining - self.progress_eps:
            self.ref_remaining = d
            self.last_progress_time = self.now()
        elif d > self.ref_remaining + 0.5:
            # replanned onto a longer route (normal in a maze): new baseline
            self.ref_remaining = d
            self.last_progress_time = self.now()

    def nav_result_cb(self, seq, future):
        if seq != self.seq:
            return
        status = future.result().status

        if self.goal_kind == 'home':
            self.get_logger().info(f'Return-home finished (status {status}).')
            self.set_done()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.n_reached += 1
            self.visited.append(self.current)
            self.get_logger().info(
                f'Reached goal ({self.n_reached}/{self.n_sent} reached).')
            if self.spin_on_arrival:
                self.start_spin()
                return
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'Goal cancelled ({self.cancel_reason}).')
        else:
            if self.cancel_reason is None:         # plain abort from Nav2
                self.failed.append(self.current)
                self.n_failed += 1
            self.get_logger().warn(f'Goal failed (status {status}); blacklisted.')
        self.go_idle()

    def cancel_current(self, reason, failed):
        if self.cancel_reason is not None:
            return
        self.cancel_reason = reason
        self.cancel_time = self.now()
        if failed and self.goal_kind == 'frontier':
            self.failed.append(self.current)
            self.n_failed += 1
            self.get_logger().warn(f'Cancelling goal: {reason}; blacklisted.')
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    # ============================================================== spin
    def start_spin(self):
        if not self.spin_client.server_is_ready():
            self.go_idle()
            return
        goal = Spin.Goal()
        goal.target_yaw = float(self.spin_angle)
        goal.time_allowance = DurationMsg(sec=int(self.spin_timeout))
        self.seq += 1
        seq = self.seq
        self.state = SPINNING
        self.goal_handle = None
        self.goal_sent_time = self.now()
        fut = self.spin_client.send_goal_async(goal)
        fut.add_done_callback(lambda f, s=seq: self.spin_response_cb(s, f))

    def spin_response_cb(self, seq, future):
        if seq != self.seq:
            return
        handle = future.result()
        if not handle.accepted:
            self.go_idle()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f, s=seq: (self.go_idle() if s == self.seq else None))

    # ============================================================ watchdog
    def watchdog(self):
        if self.state == WAITING:
            if not self.nav_client.server_is_ready():
                self.get_logger().info('Waiting for navigate_to_pose server...',
                                       throttle_duration_sec=5.0)
                return
            if self.nav_seen_time is None:
                self.nav_seen_time = self.now()
            if self.secs_since(self.nav_seen_time) >= self.startup_delay:
                self.get_logger().info('Nav2 is up - exploring.')
                self.go_idle()
            return

        if self.state == NAVIGATING:
            if self.cancel_reason is not None:
                # Nav2 should confirm the cancel quickly; don't hang forever
                if self.secs_since(self.cancel_time) > 10.0:
                    self.get_logger().warn('Cancel not confirmed; moving on.')
                    self.seq += 1                  # ignore any late result
                    if self.goal_kind == 'home':
                        self.set_done()
                    else:
                        self.go_idle()
                return
            if self.secs_since(self.goal_sent_time) > self.goal_timeout:
                self.cancel_current('timeout', failed=True)
            elif self.secs_since(self.last_progress_time) > self.progress_timeout:
                self.cancel_current('no progress', failed=True)
            return

        if self.state == SPINNING:
            if self.secs_since(self.goal_sent_time) > self.spin_timeout + 5.0:
                if self.goal_handle is not None:
                    self.goal_handle.cancel_goal_async()
                self.seq += 1
                self.go_idle()

    # ============================================================== finish
    def finish(self):
        self.get_logger().info(
            f'Exploration complete: {self.n_sent} goals sent, {self.n_reached} '
            f'reached, {self.n_failed} failed.')
        if self.return_home and not self.home_attempted:
            self.home_attempted = True
            self.get_logger().info('Returning to start pose.')
            self.send_nav_goal(self.home[0], self.home[1], 0.0, kind='home')
            return
        self.set_done()

    def set_done(self):
        self.state = DONE
        self.done_pub.publish(Bool(data=True))
        self.get_logger().info('DONE.')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
