#!/usr/bin/env python3
"""
aruco_detector.py - detects 6x6 ArUco markers and localises them in /map.

sub  /camera/image_raw        sensor_msgs/Image
sub  /camera/camera_info      sensor_msgs/CameraInfo
sub  /exploration_complete    std_msgs/Bool  (prints final summary)
tf   map -> camera optical frame
pub  /aruco_markers           visualization_msgs/MarkerArray (RViz)
file output_file (JSON)       {id: {x, y, z, n}}

For each detection: solvePnP (IPPE_SQUARE) gives the marker centre in the
camera optical frame; TF moves it into the map frame.  Estimates per ID
are the median of all detections (robust to the odd bad frame), and a
marker is only reported once seen `min_detections` times.  Detections
further than `max_range` are ignored because pose error grows with range.

Works with both old (<4.7) and new OpenCV ArUco APIs.
"""
import json
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException


def make_detector(dict_name):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, 'ArucoDetector'):                    # OpenCV >= 4.7
        det = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return det.detectMarkers
    params = aruco.DetectorParameters_create()             # OpenCV < 4.7
    return lambda img: aruco.detectMarkers(img, dictionary, parameters=params)


def image_to_gray(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    rows = buf.reshape(h, step)
    if enc in ('mono8', '8uc1'):
        return rows[:, :w].copy()
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(enc)
    if ch is None:
        return None
    img = rows[:, :w * ch].reshape(h, w, ch)
    code = {'rgb8': cv2.COLOR_RGB2GRAY, 'bgr8': cv2.COLOR_BGR2GRAY,
            'rgba8': cv2.COLOR_RGBA2GRAY, 'bgra8': cv2.COLOR_BGRA2GRAY}[enc]
    return cv2.cvtColor(img, code)


def transform_point(tf, p):
    """Apply geometry_msgs/TransformStamped to a 3-vector."""
    q = tf.transform.rotation
    t = tf.transform.translation
    qv = np.array([q.x, q.y, q.z])
    v = np.asarray(p, dtype=float)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    v = v + 2.0 * (q.w * uv + uuv)
    return v + np.array([t.x, t.y, t.z])


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        dp = self.declare_parameter
        dp('image_topic', '/camera/image_raw')
        dp('camera_info_topic', '/camera/camera_info')
        dp('map_frame', 'map')
        dp('camera_frame', '')             # '' = use image header frame_id
        dp('frame_is_optical', True)       # z forward, x right, y down
        dp('dictionary', 'DICT_6X6_250')
        dp('marker_size', 0.10)            # m (brief: 100 mm)
        dp('max_range', 3.0)               # m
        dp('min_detections', 3)
        dp('output_file', '/tmp/aruco_markers.json')

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = gp('map_frame')
        self.camera_frame = gp('camera_frame')
        self.frame_is_optical = bool(gp('frame_is_optical'))
        self.max_range = float(gp('max_range'))
        self.min_det = int(gp('min_detections'))
        self.output_file = gp('output_file')
        s = float(gp('marker_size')) / 2.0
        # corner order required by SOLVEPNP_IPPE_SQUARE (TL, TR, BR, BL)
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                                dtype=np.float64)
        self.detect = make_detector(gp('dictionary'))

        self.K = None
        self.D = None
        self.samples = {}                  # id -> list of np.array(3)
        self.reported = set()
        self.dirty = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, gp('camera_info_topic'),
                                 self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, gp('image_topic'),
                                 self.image_cb, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, 'exploration_complete', self.done_cb, latched)
        self.marker_pub = self.create_publisher(MarkerArray, 'aruco_markers', 10)
        self.create_timer(2.0, self.periodic)
        self.get_logger().info('ArUco detector started.')

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        self.D = d if d.size else np.zeros(5)

    def lookup(self, frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(self.map_frame, frame, Time.from_msg(stamp))
        except TransformException:
            pass
        try:                                   # fall back to the latest TF
            return self.tf_buffer.lookup_transform(self.map_frame, frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f'TF {self.map_frame}->{frame}: {exc}',
                                   throttle_duration_sec=5.0)
            return None

    def image_cb(self, msg):
        if self.K is None:
            self.get_logger().info('Waiting for camera_info...', throttle_duration_sec=5.0)
            return
        gray = image_to_gray(msg)
        if gray is None:
            self.get_logger().warn(f'Unsupported encoding {msg.encoding}',
                                   throttle_duration_sec=10.0)
            return
        corners, ids, _ = self.detect(gray)
        if ids is None or len(ids) == 0:
            return

        tf = self.lookup(self.camera_frame or msg.header.frame_id, msg.header.stamp)
        if tf is None:
            return

        for c, mid in zip(corners, ids.flatten()):
            ok, _rvec, tvec = cv2.solvePnP(
                self.obj_pts, c.reshape(4, 2).astype(np.float64), self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            p = tvec.flatten()
            rng = float(np.linalg.norm(p))
            if rng > self.max_range:
                continue
            if not self.frame_is_optical:      # convert to x-forward body axes
                p = np.array([p[2], -p[0], -p[1]])
            self.add_sample(int(mid), transform_point(tf, p), rng)

    def add_sample(self, mid, pos, rng):
        lst = self.samples.setdefault(mid, [])
        lst.append(pos)
        if len(lst) > 300:
            del lst[0]
        self.dirty = True
        if len(lst) >= self.min_det and mid not in self.reported:
            self.reported.add(mid)
            x, y, _ = self.estimate(mid)
            self.get_logger().info(
                f'*** Marker {mid} found at map ({x:.2f}, {y:.2f}) '
                f'[first seen at {rng:.1f} m] ***')

    def estimate(self, mid):
        return np.median(np.array(self.samples[mid]), axis=0)

    # ------------------------------------------------------------------
    def results(self):
        out = {}
        for mid in sorted(self.reported):
            x, y, z = self.estimate(mid)
            out[str(mid)] = {'x': round(float(x), 3), 'y': round(float(y), 3),
                             'z': round(float(z), 3), 'n': len(self.samples[mid])}
        return out

    def periodic(self):
        if not self.dirty:
            return
        self.dirty = False
        self.publish_markers()
        self.save()

    def save(self):
        try:
            with open(self.output_file, 'w') as fh:
                json.dump(self.results(), fh, indent=2)
        except OSError as exc:
            self.get_logger().error(f'Could not write {self.output_file}: {exc}')

    def publish_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for mid in sorted(self.reported):
            x, y, z = self.estimate(mid)
            cube = Marker()
            cube.header.frame_id = self.map_frame
            cube.header.stamp = stamp
            cube.ns, cube.id = 'aruco', mid
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = Point(x=float(x), y=float(y), z=float(z))
            cube.pose.orientation.w = 1.0
            cube.scale.x = cube.scale.y = cube.scale.z = 0.15
            cube.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.9)
            arr.markers.append(cube)

            txt = Marker()
            txt.header = cube.header
            txt.ns, txt.id = 'aruco_label', mid
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position = Point(x=float(x), y=float(y), z=float(z) + 0.3)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.2
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f'ID {mid}'
            arr.markers.append(txt)
        self.marker_pub.publish(arr)

    def done_cb(self, msg):
        if msg.data:
            self.summary()

    def summary(self):
        self.save()
        res = self.results()
        lines = [f'  ID {k}: ({v["x"]:.2f}, {v["y"]:.2f})  n={v["n"]}' for k, v in res.items()]
        self.get_logger().info(
            f'{len(res)} marker(s) localised (saved to {self.output_file}):\n'
            + ('\n'.join(lines) if lines else '  none'))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
