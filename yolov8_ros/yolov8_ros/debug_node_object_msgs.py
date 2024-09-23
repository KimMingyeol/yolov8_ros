# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

from object_msgs.msg import Object
from object_msgs.msg import ObjectInBox
from object_msgs.msg import ObjectsInBoxes


class DebugNodeObjectMsgs(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("debug_node_object_msgs")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile)
        self.detections_sub = message_filters.Subscriber(
            self, ObjectsInBoxes, "detections", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._dbg_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def draw_box(self, cv_image: np.ndarray, detection: ObjectInBox, color: Tuple[int]) -> np.ndarray:

        # get detection info
        label = detection.object.object_name
        score = detection.object.probability
        box_msg: RegionOfInterest = detection.roi

        min_pt = (round(box_msg.x_offset),
                  round(box_msg.y_offset))
        max_pt = (round(box_msg.x_offset + box_msg.width),
                  round(box_msg.y_offset + box_msg.height))

        # define the four corners of the rectangle
        rect_pts = np.array([
            [min_pt[0], min_pt[1]],
            [max_pt[0], min_pt[1]],
            [max_pt[0], max_pt[1]],
            [min_pt[0], max_pt[1]]
        ])

        # calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            (box_msg.x_offset + box_msg.width/2, box_msg.y_offset + box_msg.height/2),
            0.0,
            1.0
        )

        # rotate the corners of the rectangle
        rect_pts = np.int0(cv2.transform(
            np.array([rect_pts]), rotation_matrix)[0])

        # Draw the rotated rectangle
        for i in range(4):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % 4])
            cv2.line(cv_image, pt1, pt2, color, 2)

        # write text
        label = "{} ({:.3f})".format(label, score)
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font,
                    1, color, 1, cv2.LINE_AA)

        return cv_image

    def detections_cb(self, img_msg: Image, detection_msg: ObjectsInBoxes) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        detection: ObjectInBox
        for detection in detection_msg.objects_vector:

            # random color
            label = detection.object.object_name

            if label not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[label] = (r, g, b)

            color = self._class_to_color[label]

            cv_image = self.draw_box(cv_image, detection, color)

        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))


def main():
    rclpy.init()
    node = DebugNodeObjectMsgs()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
