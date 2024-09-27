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


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    #
    # ARGS
    #
    model_type = LaunchConfiguration("model_type")
    model_type_cmd = DeclareLaunchArgument(
        "model_type",
        default_value="YOLO",
        choices=["YOLO", "NAS"],
        description="Model type form Ultralytics (YOLO, NAS")

    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="yolov8m.pt",
        description="Model name or path")

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)")
    
    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.5",
        description="Minimum probability of a detection to be published")

    input_image_topic = LaunchConfiguration("input_image_topic")
    input_image_topic_cmd = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/cubemap/front",
        description="Name of the input image topic")

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes")

    #
    # NODES
    #
    detector_node_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_node_object_msgs",
        name="yolov8_node_object_msgs",
        namespace=namespace,
        parameters=[{
            "model_type": model_type,
            "model": model,
            "device": device,
            "threshold": threshold,
        }],
        remappings=[("image_raw", input_image_topic)]
    )

    debug_node_cmd = Node(
        package="yolov8_ros",
        executable="debug_node_object_msgs",
        name="debug_node_object_msgs",
        namespace=namespace,
        remappings=[
            ("image_raw", input_image_topic),
            ("detections", "detections_object_msgs")
        ]
    )

    ld = LaunchDescription()

    ld.add_action(model_type_cmd)
    ld.add_action(model_cmd)
    ld.add_action(device_cmd)
    ld.add_action(threshold_cmd)
    ld.add_action(input_image_topic_cmd)
    ld.add_action(namespace_cmd)

    ld.add_action(detector_node_cmd)
    # ld.add_action(debug_node_cmd)

    return ld
