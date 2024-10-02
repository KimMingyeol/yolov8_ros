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


from typing import List, Dict

import rclpy
from rclpy.qos import QoSProfile, QoSPresetProfiles
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from cv_bridge import CvBridge

import torch
from ultralytics import YOLO, NAS
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes

from sensor_msgs.msg import Image

from object_msgs.msg import Object
from object_msgs.msg import ObjectInBox
from object_msgs.msg import ObjectsInBoxes


class Yolov8Node(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolov8_node_object_msgs")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("threshold", 0.5)

        self.type_to_model = {
            "YOLO": YOLO,
            "NAS": NAS
        }

        self.detection_class_names = ["person"]
        self.detection_class_ids = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.model_type = self.get_parameter(
            "model_type").get_parameter_value().string_value

        self.model = self.get_parameter(
            "model").get_parameter_value().string_value

        self.device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self._pub_object_msgs =  self.create_lifecycle_publisher(
            ObjectsInBoxes, "detections_object_msgs", 10)
        
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        self.yolo = self.type_to_model[self.model_type](self.model)
        self.yolo.fuse()

        self.detection_class_ids = list(map(lambda x: [k for k, v in self.yolo.names.items() if v==x][0], self.detection_class_names))

        # subs
        self._sub = self.create_subscription(
            Image,
            "image_raw",
            self.image_cb_object_msgs,
            10
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub_object_msgs)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf)
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i])
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list
    
    def parse_boxes_dict(self, results: Results) -> List[Dict]:

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                # get boxes values
                box = box_data.xywh[0]
                center_position_x = float(box[0])
                center_position_y = float(box[1])
                size_x = float(box[2])
                size_y = float(box[3])

                boxes_list.append({"x_offset": center_position_x - size_x/2,"y_offset": center_position_y - size_y/2, "width":size_x, "height":size_y})

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                # get boxes values
                box = results.obb.xywhr[i]
                center_position_x = float(box[0])
                center_position_y = float(box[1])
                center_theta = float(box[4])
                size_x = float(box[2])
                size_y = float(box[3])
                
                boxes_list.append({"x_offset": center_position_x - size_x/2,"y_offset": center_position_y - size_y/2, "width":size_x, "height":size_y})

        return boxes_list

    def image_cb_object_msgs(self, msg:Image) -> None:
        # convert image + predict
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device,
            classes=self.detection_class_ids
        )
        results: Results = results[0].cpu()

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes_dict(results)

        # create detection msgs
        detections_msg = ObjectsInBoxes()

        for i in range(len(results)):
            aux_msg = ObjectInBox()

            if results.boxes or results.obb and hypothesis and boxes:
                aux_msg_obj = Object()
                aux_msg_obj.object_name = hypothesis[i]["class_name"]
                aux_msg_obj.probability = hypothesis[i]["score"]

                aux_msg.object = aux_msg_obj
                aux_msg.roi.x_offset = round(boxes[i]["x_offset"])
                aux_msg.roi.y_offset = round(boxes[i]["y_offset"])
                aux_msg.roi.width = round(boxes[i]["width"])
                aux_msg.roi.height = round(boxes[i]["height"])

            detections_msg.objects_vector.append(aux_msg)
        
        # publish detections
        detections_msg.header = msg.header
        self._pub_object_msgs.publish(detections_msg)

        del results
        del cv_image


def main():
    rclpy.init()
    node = Yolov8Node()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
