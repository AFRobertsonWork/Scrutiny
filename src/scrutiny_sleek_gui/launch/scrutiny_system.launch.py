from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("zed_wrapper"),
                "launch",
                "zed_camera.launch.py",
            )
        ),
        launch_arguments={"camera_model": "zedx"}.items(),
    )

    tof_node = Node(
        package="scrutiny_tof_driver",
        executable="scrutiny_tof_node",
        name="scrutiny_tof_node",
        output="screen",
    )

    measurement_node = Node(
        package="scrutiny_measurement",
        executable="fusion_node",
        name="scrutiny_fusion_node",
        output="screen",
    )

    gui_node = Node(
        package="scrutiny_sleek_gui",
        executable="scrutiny_gui",
        name="scrutiny_gui",
        output="screen",
        parameters=[
            {"zed_left_topic": "/zed/zed_node/left/color/rect/image"},
            {"zed_depth_topic": "/zed/zed_node/depth/depth_registered"},
            {"tof_topic": "/scrutiny/tof/depth8"},
            {"measurement_topic": "/scrutiny/measurement/fused_overlay"},
        ],
    )

    return LaunchDescription([
        zed_launch,
        tof_node,
        measurement_node,
        gui_node,
    ])
