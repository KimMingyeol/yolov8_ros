import rclpy
from rclpy.qos import QoSProfile, QoSPresetProfiles
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from geometry_msgs.msg import Point32, Pose, Vector3, Point, Quaternion
from sensor_msgs.msg import RegionOfInterest
import std_msgs.msg as std_msgs

from object_msgs.msg import Object
from object_analytics_msgs.msg import ObjectInBox3D
from object_analytics_msgs.msg import ObjectsInBoxes3D

from vision_msgs.msg import BoundingBox3D
from vision_msgs.msg import BoundingBox3DArray

class BoundingBox3dRvizPublisher(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("bounding_box_3d_rviz_publisher")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # pubs
        self._pub = self.create_lifecycle_publisher(BoundingBox3DArray, "bounding_box_3d_rviz", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self._sub = self.create_subscription(
            ObjectsInBoxes3D,
            "detections_3d",
            self.detections_cb,
            10
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def detections_cb(self, msg: ObjectsInBoxes3D) -> None:

        bounding_box_3d_array = BoundingBox3DArray()

        object_in_box_3d: ObjectInBox3D
        for object_in_box_3d in msg.objects_in_boxes:
            bounding_box_3d = BoundingBox3D()

            # obj_name: str = object_in_box_3d.object.object_name # string
            # prob: float = object_in_box_3d.object.probability # float32
            # region_of_interest: RegionOfInterest = object_in_box_3d.roi # RegionOfInterest
            point_min: Point32 = object_in_box_3d.min # Point32
            point_max: Point32 = object_in_box_3d.max # Point32

            center_bounding_box_3d = Pose()
            center_bounding_box_3d.position.x = (point_max.x + point_min.x) / 2
            center_bounding_box_3d.position.y = (point_max.y + point_min.y) / 2
            center_bounding_box_3d.position.z = (point_max.z + point_min.z) / 2
            center_bounding_box_3d.orientation.x = 0. ### need to modify
            center_bounding_box_3d.orientation.y = 0.
            center_bounding_box_3d.orientation.z = 0.
            center_bounding_box_3d.orientation.w = 1.

            size_bounding_box_3d = Vector3()
            size_bounding_box_3d.x = point_max.x - point_min.x
            size_bounding_box_3d.y = point_max.y - point_min.y
            size_bounding_box_3d.z = point_max.z - point_min.z

            bounding_box_3d.center = center_bounding_box_3d
            bounding_box_3d.size = size_bounding_box_3d

            bounding_box_3d_array.boxes.append(bounding_box_3d)
        
        bounding_box_3d_array.header = std_msgs.Header(stamp=msg.header.stamp, frame_id='livox')
        self._pub.publish(bounding_box_3d_array)


def main():
    rclpy.init()
    node = BoundingBox3dRvizPublisher()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
