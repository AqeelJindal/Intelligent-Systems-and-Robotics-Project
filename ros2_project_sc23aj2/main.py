#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image

from math import cos, sin


# ==============================================================
# CONSTANT SETTINGS
# ==============================================================

WINDOW_TITLE = 'Colour Object Tracker'
CAMERA_TOPIC = '/camera/image_raw'
VELOCITY_TOPIC = '/cmd_vel'
NAV_ACTION_NAME = 'navigate_to_pose'
MAP_FRAME = 'map'

WAYPOINTS = [
    (4.16, -1.76, -0.00143),
    (-0.5, -4.36, 0.00247),
    (3.91, -9.03, 0.00247),
]

HSV_LIMITS = {
    'red': {
        'lower': np.array([0, 150, 50]),
        'upper': np.array([10, 255, 255]),
        'lower_wrap': np.array([170, 150, 50]),
        'upper_wrap': np.array([180, 255, 255]),
        'draw_colour': (0, 0, 255),
        'display_name': 'Red',
    },
    'green': {
        'lower': np.array([40, 70, 70]),
        'upper': np.array([80, 255, 255]),
        'draw_colour': (0, 255, 0),
        'display_name': 'Green',
    },
    'blue': {
        'lower': np.array([100, 150, 50]),
        'upper': np.array([140, 255, 255]),
        'draw_colour': (255, 0, 0),
        'display_name': 'Blue',
    },
}


def clamp(value, lower_limit, upper_limit):
    """Restrict a numeric value to the given minimum and maximum range."""
    return max(min(value, upper_limit), lower_limit)


class ColourSearchNavigator(Node):
    """ROS2 node for searching coloured boxes and approaching the blue target."""

    def __init__(self):
        super().__init__('colour_search_navigator')

        self.cv_bridge = CvBridge()
        self.camera_listener = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.camera_frame_callback,
            10,
        )
        self.nav_client = ActionClient(self, NavigateToPose, NAV_ACTION_NAME)
        self.velocity_publisher = self.create_publisher(Twist, VELOCITY_TOPIC, 10)

        self.seen_colour = {
            'red': False,
            'green': False,
            'blue': False,
        }

        self.target_visible = False
        self.target_pixel_x = None
        self.target_size = 0.0

        self.frame_width = 960
        self.frame_middle_x = self.frame_width // 2

        self.arrival_area_limit = 270000
        self.target_alignment_tolerance = 25
        self.minimum_blob_area = 1000
        self.navigation_cancel_area = 5000
        self.scan_cancel_area = 2000

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 800, 600)

        self.get_logger().info('Colour search navigator has started.')
        self.get_logger().info('Vision tracking, waypoint navigation, scanning and target approach are active.')

    # ==========================================================
    # CAMERA PROCESSING
    # ==========================================================

    def camera_frame_callback(self, image_msg):
        """Process one camera frame and update the current target information."""
        frame = self._ros_image_to_cv_image(image_msg)
        if frame is None:
            return

        self._update_frame_geometry(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self._clear_current_target_reading()
        self._detect_and_draw_colours(frame, hsv_frame)
        self._draw_tracking_summary(frame)

        cv2.imshow(WINDOW_TITLE, frame)
        cv2.waitKey(1)

    def _ros_image_to_cv_image(self, image_msg):
        """Convert a ROS image message into an OpenCV BGR image."""
        try:
            return self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().error(f'Camera conversion failed: {error}')
            return None

    def _update_frame_geometry(self, frame):
        """Store the current camera width and horizontal centre point."""
        _, width = frame.shape[:2]
        self.frame_width = width
        self.frame_middle_x = width // 2

    def _clear_current_target_reading(self):
        """Reset only the blue target values for the current frame."""
        self.target_visible = False
        self.target_pixel_x = None
        self.target_size = 0.0

    def _detect_and_draw_colours(self, frame, hsv_frame):
        """Detect each colour range and annotate visible objects on the frame."""
        for colour_key, colour_config in HSV_LIMITS.items():
            mask = self._make_hsv_mask(hsv_frame, colour_key, colour_config)
            blob = self._largest_valid_blob(mask)

            if blob is None:
                continue

            x_pos, y_pos, width, height, area, centre_x, centre_y = blob
            self._record_colour_detection(colour_key, centre_x, area)
            self._draw_blob_annotation(frame, colour_config, x_pos, y_pos, width, height, centre_x, centre_y, area)

    def _make_hsv_mask(self, hsv_frame, colour_key, colour_config):
        """Create a cleaned HSV mask for a colour range."""
        if colour_key == 'red':
            first_mask = cv2.inRange(hsv_frame, colour_config['lower'], colour_config['upper'])
            second_mask = cv2.inRange(hsv_frame, colour_config['lower_wrap'], colour_config['upper_wrap'])
            colour_mask = cv2.bitwise_or(first_mask, second_mask)
        else:
            colour_mask = cv2.inRange(hsv_frame, colour_config['lower'], colour_config['upper'])

        clean_kernel = np.ones((5, 5), np.uint8)
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN, clean_kernel)
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, clean_kernel)
        return colour_mask

    def _largest_valid_blob(self, colour_mask):
        """Return the position, size and centre of the largest valid contour."""
        contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)

        if contour_area < self.minimum_blob_area:
            return None

        x_pos, y_pos, width, height = cv2.boundingRect(main_contour)
        contour_moments = cv2.moments(main_contour)

        if contour_moments['m00'] == 0:
            return None

        centre_x = int(contour_moments['m10'] / contour_moments['m00'])
        centre_y = int(contour_moments['m01'] / contour_moments['m00'])
        return x_pos, y_pos, width, height, contour_area, centre_x, centre_y

    def _record_colour_detection(self, colour_key, centre_x, area):
        """Update stored state when a coloured object is detected."""
        self.seen_colour[colour_key] = True

        if colour_key == 'blue':
            self.target_visible = True
            self.target_pixel_x = centre_x
            self.target_size = area

    def _draw_blob_annotation(self, frame, colour_config, x_pos, y_pos, width, height, centre_x, centre_y, area):
        """Draw the detection rectangle, centre dot and text label."""
        draw_colour = colour_config['draw_colour']
        label = f"{colour_config['display_name']}: {int(area)}"

        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y_pos + height), draw_colour, 2)
        cv2.circle(frame, (centre_x, centre_y), 5, draw_colour, -1)
        cv2.putText(
            frame,
            label,
            (x_pos, max(y_pos - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_colour,
            2,
        )

    def _draw_tracking_summary(self, frame):
        """Show which colours have been seen so far."""
        summary_text = (
            f"Seen: Red={self.seen_colour['red']} "
            f"Green={self.seen_colour['green']} "
            f"Blue={self.seen_colour['blue']}"
        )
        cv2.putText(
            frame,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # ==========================================================
    # WAYPOINT EXPLORATION
    # ==========================================================

    def move_to_waypoint(self, x_position, y_position, yaw_angle):
        """Navigate to one waypoint and cancel if the blue target becomes visible."""
        target_goal = self._build_navigation_goal(x_position, y_position, yaw_angle)

        self.get_logger().info(
            f'Navigating to waypoint: x={x_position:.2f}, y={y_position:.2f}, yaw={yaw_angle:.2f}'
        )
        self.nav_client.wait_for_server()

        send_future = self.nav_client.send_goal_async(target_goal)
        rclpy.spin_until_future_complete(self, send_future)

        active_goal = send_future.result()
        if active_goal is None or not active_goal.accepted:
            self.get_logger().warn('Nav2 rejected the waypoint goal.')
            return False

        self.get_logger().info('Waypoint goal accepted.')
        result_future = active_goal.get_result_async()

        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

            if self._target_is_large_enough(self.navigation_cancel_area):
                self.get_logger().info('Blue target detected during navigation. Cancelling waypoint goal.')
                cancel_future = active_goal.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                self.halt_motion()
                return True

        self.get_logger().info('Waypoint navigation finished.')
        return False

    def _build_navigation_goal(self, x_position, y_position, yaw_angle):
        """Create a NavigateToPose goal message from map coordinates."""
        target_goal = NavigateToPose.Goal()
        target_goal.pose.header.frame_id = MAP_FRAME
        target_goal.pose.header.stamp = self.get_clock().now().to_msg()

        target_goal.pose.pose.position.x = float(x_position)
        target_goal.pose.pose.position.y = float(y_position)
        target_goal.pose.pose.position.z = 0.0

        target_goal.pose.pose.orientation.z = sin(yaw_angle / 2.0)
        target_goal.pose.pose.orientation.w = cos(yaw_angle / 2.0)
        return target_goal

    def scan_from_current_position(self):
        """Sweep left and right from the current pose to search for the blue target."""
        self.get_logger().info('Starting left-right sweep scan.')

        sweep_speed = 0.35
        sweep_pattern = [
            (sweep_speed, 2.0, 'Sweeping left.'),
            (0.0, 0.4, 'Checking after left sweep.'),
            (-sweep_speed, 4.0, 'Sweeping right.'),
            (0.0, 0.4, 'Checking after right sweep.'),
            (sweep_speed, 2.0, 'Returning towards centre.'),
            (0.0, 0.4, 'Final sweep check.'),
        ]

        for angular_speed, duration, message in sweep_pattern:
            self.get_logger().info(message)

            target_found = self._run_timed_scan_motion(
                angular_speed=angular_speed,
                duration=duration,
            )

            if target_found:
                self.get_logger().info('Blue target found during left-right sweep scan.')
                return True

        self.halt_motion()
        self.get_logger().info('Left-right sweep scan finished.')
        return False

    def _run_timed_scan_motion(self, angular_speed, duration):
        """Publish a timed turning command while still checking camera frames."""
        scan_command = Twist()
        scan_command.angular.z = angular_speed

        motion_start = self._seconds_now()

        while rclpy.ok() and self._seconds_now() - motion_start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self._target_is_large_enough(self.scan_cancel_area):
                self.halt_motion()
                return True

            self.velocity_publisher.publish(scan_command)

        self.halt_motion()
        return False

    def _seconds_now(self):
        """Return the current ROS clock time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def _target_is_large_enough(self, area_limit):
        """Check whether the blue target is visible and exceeds an area threshold."""
        return self.target_visible and self.target_size > area_limit

    # ==========================================================
    # BLUE TARGET APPROACH
    # ==========================================================

    def drive_towards_blue_target(self):
        """Approach the blue object while keeping it horizontally centred."""
        self.get_logger().info('Blue target approach started.')

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            drive_command = Twist()

            if not self.target_visible or self.target_pixel_x is None:
                drive_command.angular.z = 0.25
                self.velocity_publisher.publish(drive_command)
                continue

            horizontal_error = self.frame_middle_x - self.target_pixel_x
            target_is_aligned = abs(horizontal_error) < self.target_alignment_tolerance

            if self.target_size >= self.arrival_area_limit and target_is_aligned:
                self.get_logger().info('Blue target reached. Stopping robot.')
                self.halt_motion()
                return True

            if self.target_size < self.arrival_area_limit:
                drive_command.linear.x = 0.12
            else:
                drive_command.linear.x = 0.03

            drive_command.angular.z = float(horizontal_error) * 0.007
            drive_command.linear.x = clamp(drive_command.linear.x, -0.15, 0.15)
            drive_command.angular.z = clamp(drive_command.angular.z, -0.6, 0.6)

            self.velocity_publisher.publish(drive_command)

    def halt_motion(self):
        """Publish a zero Twist command to stop the robot."""
        self.velocity_publisher.publish(Twist())


# ==============================================================
# PROGRAM ENTRY POINT
# ==============================================================


def warm_up_camera(node, frame_count=10):
    """Allow a few camera callbacks to run before the robot starts moving."""
    for _ in range(frame_count):
        rclpy.spin_once(node, timeout_sec=0.1)


def run_search_route(node, route):
    """Visit each waypoint, scan locally, and approach the blue object when found."""
    for x_position, y_position, yaw_angle in route:
        if not rclpy.ok():
            return

        found_while_moving = node.move_to_waypoint(x_position, y_position, yaw_angle)
        if found_while_moving or node.target_visible:
            node.drive_towards_blue_target()
            return

        found_while_scanning = node.scan_from_current_position()
        if found_while_scanning or node.target_visible:
            node.drive_towards_blue_target()
            return

    node.get_logger().info('All waypoints searched. Blue target was not found.')


def main(args=None):
    rclpy.init(args=args)
    colour_node = ColourSearchNavigator()

    try:
        warm_up_camera(colour_node)
        run_search_route(colour_node, WAYPOINTS)

    except KeyboardInterrupt:
        colour_node.get_logger().info('Keyboard interrupt received. Stopping robot.')

    finally:
        colour_node.halt_motion()
        cv2.destroyAllWindows()
        colour_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

