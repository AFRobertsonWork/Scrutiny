#!/usr/bin/env python3

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32


class FusionNode(Node):

    def __init__(self):
        super().__init__("scrutiny_fusion_node")

        self.bridge = CvBridge()

        self.declare_parameter(
            "calibration_file",
            str(
                Path.home()
                / "scrutiny_ws"
                / "src"
                / "scrutiny_calibration"
                / "config"
                / "zed_tof_calibration.yaml"
            ),
        )

        self.declare_parameter(
            "rgb_topic",
            "/zed/zed_node/left/color/rect/image",
        )

        self.declare_parameter(
            "zed_depth_topic",
            "/zed/zed_node/depth/depth_registered",
        )

        self.declare_parameter(
            "tof_depth_topic",
            "/scrutiny/tof/depth_raw",
        )

        self.declare_parameter(
            "zed_camera_info_topic",
            "/zed/zed_node/left/color/rect/camera_info",
        )

        self.declare_parameter(
            "tof_camera_info_topic",
            "/scrutiny/tof/camera_info",
        )

        self.calibration_file = Path(
            str(
                self.get_parameter(
                    "calibration_file"
                ).value
            )
        )

        self.rgb = None
        self.zed_depth = None
        self.tof_depth = None

        self.zed_K = None
        self.zed_D = None
        self.tof_K = None
        self.tof_D = None

        self.R = None
        self.T = None

        self.history = deque(maxlen=10)

        self.load_calibration()

        self.create_subscription(
            Image,
            self.get_parameter("rgb_topic").value,
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            self.get_parameter("zed_depth_topic").value,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            self.get_parameter("tof_depth_topic").value,
            self.tof_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter(
                "zed_camera_info_topic"
            ).value,
            self.zed_info_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter(
                "tof_camera_info_topic"
            ).value,
            self.tof_info_callback,
            qos_profile_sensor_data,
        )

        self.publisher = self.create_publisher(
            Float32,
            "/scrutiny/measurement/diameter_mm",
            5,
        )

        self.overlay_publisher = self.create_publisher(
            Image,
            "/scrutiny/measurement/fused_overlay",
            5,
        )

        self.timer = self.create_timer(
            0.1,
            self.process,
        )

        self.get_logger().info(
            "Fusion node started."
        )

    def load_calibration(self):
        if not self.calibration_file.exists():
            raise FileNotFoundError(
                f"Calibration file not found: "
                f"{self.calibration_file}"
            )

        with self.calibration_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        self.R = np.asarray(
            data["rotation_matrix"],
            dtype=np.float64,
        ).reshape(3, 3)

        self.T = np.asarray(
            data["translation_m"],
            dtype=np.float64,
        ).reshape(3)

    def rgb_callback(self, msg):
        try:
            self.rgb = np.asarray(
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )
            ).copy()
        except Exception as error:
            self.get_logger().error(
                f"RGB conversion failed: {error}"
            )

    def depth_callback(self, msg):
        try:
            self.zed_depth = np.asarray(
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )
            ).copy()
        except Exception as error:
            self.get_logger().error(
                f"ZED depth conversion failed: {error}"
            )

    def tof_callback(self, msg):
        try:
            self.tof_depth = np.asarray(
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )
            ).copy()
        except Exception as error:
            self.get_logger().error(
                f"ToF depth conversion failed: {error}"
            )

    def zed_info_callback(self, msg):
        self.zed_K = np.asarray(
            msg.k,
            dtype=np.float64,
        ).reshape(3, 3)

        self.zed_D = np.asarray(
            msg.d,
            dtype=np.float64,
        )

    def tof_info_callback(self, msg):
        self.tof_K = np.asarray(
            msg.k,
            dtype=np.float64,
        ).reshape(3, 3)

        self.tof_D = np.asarray(
            msg.d,
            dtype=np.float64,
        )

    @staticmethod
    def depth_to_m(depth):
        result = depth.astype(
            np.float32
        ).copy()

        valid = (
            np.isfinite(result)
            & (result > 0)
        )

        if not np.any(valid):
            return result

        if float(
            np.median(
                result[valid]
            )
        ) > 20.0:
            result /= 1000.0

        return result

    @staticmethod
    def stencil_geometry(shape):
        height, width = shape[:2]

        centre = (
            width // 2,
            height // 2,
        )

        axes = (
            int(width * 0.21),
            int(height * 0.28),
        )

        return centre, axes

    def create_stencil_mask(self, shape):
        mask = np.zeros(
            shape[:2],
            dtype=np.uint8,
        )

        centre, axes = self.stencil_geometry(
            shape
        )

        cv2.ellipse(
            mask,
            centre,
            axes,
            0,
            0,
            360,
            255,
            -1,
        )

        return mask

    def segment_object(self):
        if self.rgb is None:
            return None

        if self.zed_depth is None:
            return None

        depth = self.depth_to_m(
            self.zed_depth
        )

        if (
            depth.shape[:2]
            != self.rgb.shape[:2]
        ):
            depth = cv2.resize(
                depth,
                (
                    self.rgb.shape[1],
                    self.rgb.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        stencil = self.create_stencil_mask(
            self.rgb.shape
        )

        height, width = depth.shape

        centre_x = width // 2
        centre_y = height // 2

        half_width = 30
        half_height = 30

        x1 = max(
            0,
            centre_x - half_width,
        )
        x2 = min(
            width,
            centre_x + half_width,
        )
        y1 = max(
            0,
            centre_y - half_height,
        )
        y2 = min(
            height,
            centre_y + half_height,
        )

        centre_depths = depth[
            y1:y2,
            x1:x2,
        ]

        valid_centre = centre_depths[
            np.isfinite(centre_depths)
            & (centre_depths > 0.15)
            & (centre_depths < 3.0)
        ]

        if valid_centre.size < 20:
            return None

        reference = float(
            np.median(
                valid_centre
            )
        )

        valid = (
            np.isfinite(depth)
            & (depth > 0.15)
            & (depth < 3.0)
            & (stencil > 0)
        )

        mask = (
            valid
            & (
                np.abs(
                    depth - reference
                )
                < 0.07
            )
        ).astype(np.uint8) * 255

        open_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            )
        )

        close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 11),
            )
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2,
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        if count <= 1:
            return None

        best_label = None
        best_distance = float("inf")

        for label in range(
            1,
            count,
        ):
            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < 250:
                continue

            component_x = float(
                centroids[label][0]
            )
            component_y = float(
                centroids[label][1]
            )

            distance = float(
                np.hypot(
                    component_x - centre_x,
                    component_y - centre_y,
                )
            )

            if distance < best_distance:
                best_distance = distance
                best_label = label

        if best_label is None:
            return None

        object_mask = (
            labels == best_label
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if cv2.contourArea(
            contour
        ) < 250:
            return None

        return (
            object_mask,
            contour,
            depth,
        )

    def contour_to_points(
        self,
        mask,
        depth,
    ):
        if self.zed_K is None:
            return None

        rows, columns = np.where(
            (mask > 0)
            & np.isfinite(depth)
            & (depth > 0.15)
            & (depth < 3.0)
        )

        if rows.size < 100:
            return None

        z = depth[
            rows,
            columns,
        ].astype(np.float64)

        fx = float(
            self.zed_K[0, 0]
        )
        fy = float(
            self.zed_K[1, 1]
        )
        cx = float(
            self.zed_K[0, 2]
        )
        cy = float(
            self.zed_K[1, 2]
        )

        x = (
            columns.astype(np.float64)
            - cx
        ) * z / fx

        y = (
            rows.astype(np.float64)
            - cy
        ) * z / fy

        return np.column_stack(
            (
                x,
                y,
                z,
            )
        )

    def calculate_diameter(
        self,
        points,
    ):
        if points is None:
            return None

        if points.shape[0] < 100:
            return None

        median_depth = float(
            np.median(
                points[:, 2]
            )
        )

        depth_error = np.abs(
            points[:, 2]
            - median_depth
        )

        depth_limit = float(
            np.percentile(
                depth_error,
                90,
            )
        )

        points = points[
            depth_error <= depth_limit
        ]

        if points.shape[0] < 100:
            return None

        x_positions = points[:, 0]

        left_position = float(
            np.percentile(
                x_positions,
                2,
            )
        )

        right_position = float(
            np.percentile(
                x_positions,
                98,
            )
        )

        diameter_mm = (
            right_position
            - left_position
        ) * 1000.0

        if (
            not np.isfinite(
                diameter_mm
            )
            or diameter_mm < 10.0
            or diameter_mm > 500.0
        ):
            self.history.clear()
            return None

        self.history.append(
            float(
                diameter_mm
            )
        )

        return float(
            np.median(
                np.asarray(
                    self.history,
                    dtype=np.float64,
                )
            )
        )

    def draw_stencil(
        self,
        image,
        valid,
    ):
        centre, axes = (
            self.stencil_geometry(
                image.shape
            )
        )

        colour = (
            (0, 255, 255)
            if valid
            else (0, 165, 255)
        )

        cv2.ellipse(
            image,
            centre,
            axes,
            0,
            0,
            360,
            colour,
            3,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (
                centre[0] - 18,
                centre[1],
            ),
            (
                centre[0] + 18,
                centre[1],
            ),
            colour,
            2,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            (
                centre[0],
                centre[1] - 18,
            ),
            (
                centre[0],
                centre[1] + 18,
            ),
            colour,
            2,
            cv2.LINE_AA,
        )

    def draw_result(
        self,
        image,
        mask,
        contour,
        diameter_mm,
    ):
        smooth_mask = cv2.GaussianBlur(
            mask,
            (9, 9),
            0,
        )

        _, smooth_mask = cv2.threshold(
            smooth_mask,
            100,
            255,
            cv2.THRESH_BINARY,
        )

        close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 13),
            )
        )

        smooth_mask = cv2.morphologyEx(
            smooth_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            smooth_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            contour = max(
                contours,
                key=cv2.contourArea,
            )

        hull = cv2.convexHull(
            contour
        )

        cv2.drawContours(
            image,
            [hull],
            -1,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        rows, columns = np.where(
            smooth_mask > 0
        )

        if rows.size > 0:
            best_row = None
            best_left = None
            best_right = None
            best_width = -1

            for row in np.unique(
                rows
            ):
                row_columns = columns[
                    rows == row
                ]

                if row_columns.size < 2:
                    continue

                left = int(
                    np.percentile(
                        row_columns,
                        2,
                    )
                )

                right = int(
                    np.percentile(
                        row_columns,
                        98,
                    )
                )

                width = right - left

                if width > best_width:
                    best_width = width
                    best_row = int(row)
                    best_left = left
                    best_right = right

            if (
                best_row is not None
                and best_left is not None
                and best_right is not None
            ):
                point_1 = (
                    best_left,
                    best_row,
                )

                point_2 = (
                    best_right,
                    best_row,
                )

                cv2.line(
                    image,
                    point_1,
                    point_2,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    image,
                    point_1,
                    6,
                    (0, 0, 255),
                    -1,
                )

                cv2.circle(
                    image,
                    point_2,
                    6,
                    (0, 0, 255),
                    -1,
                )

                text = (
                    f"{diameter_mm:.1f} mm"
                )

                text_size, _ = (
                    cv2.getTextSize(
                        text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        3,
                    )
                )

                text_x = max(
                    5,
                    int(
                        (
                            best_left
                            + best_right
                        )
                        / 2
                        - text_size[0] / 2
                    ),
                )

                text_y = max(
                    40,
                    best_row - 25,
                )

                cv2.putText(
                    image,
                    text,
                    (
                        text_x,
                        text_y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    6,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    image,
                    text,
                    (
                        text_x,
                        text_y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        self.draw_stencil(
            image,
            True,
        )

        label = (
            f"Diameter: "
            f"{diameter_mm:.1f} mm"
        )

        cv2.putText(
            image,
            label,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            label,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def publish_overlay(
        self,
        output,
    ):
        overlay_msg = (
            self.bridge.cv2_to_imgmsg(
                output,
                encoding="bgr8",
            )
        )

        overlay_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        overlay_msg.header.frame_id = (
            "zed_left_camera_frame_optical"
        )

        self.overlay_publisher.publish(
            overlay_msg
        )

    def process(self):
        if self.rgb is None:
            return

        output = self.rgb.copy()

        if (
            self.zed_depth is None
            or self.zed_K is None
        ):
            self.draw_stencil(
                output,
                False,
            )

            cv2.putText(
                output,
                "Waiting for ZED depth",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            self.publish_overlay(
                output
            )

            cv2.imshow(
                "Scrutiny ZED Measurement",
                output,
            )

            cv2.waitKey(1)
            return

        result = self.segment_object()

        if result is None:
            self.history.clear()

            self.draw_stencil(
                output,
                False,
            )

            cv2.putText(
                output,
                "Align object on centre crosshair",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            self.publish_overlay(
                output
            )

            cv2.imshow(
                "Scrutiny ZED Measurement",
                output,
            )

            cv2.waitKey(1)
            return

        mask, contour, depth = result

        points = self.contour_to_points(
            mask,
            depth,
        )

        diameter_mm = (
            self.calculate_diameter(
                points
            )
        )

        if diameter_mm is None:
            self.history.clear()

            self.draw_stencil(
                output,
                False,
            )

            cv2.drawContours(
                output,
                [contour],
                -1,
                (0, 165, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                "Object detected - unable to measure",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            self.publish_overlay(
                output
            )

            cv2.imshow(
                "Scrutiny ZED Measurement",
                output,
            )

            cv2.waitKey(1)
            return

        self.draw_result(
            output,
            mask,
            contour,
            diameter_mm,
        )

        message = Float32()
        message.data = float(
            diameter_mm
        )

        self.publisher.publish(
            message
        )

        self.publish_overlay(
            output
        )

        cv2.imshow(
            "Scrutiny ZED Measurement",
            output,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q"),
        ):
            raise SystemExit


def main(args=None):
    rclpy.init(
        args=args
    )

    node = FusionNode()

    try:
        rclpy.spin(
            node
        )

    except (
        KeyboardInterrupt,
        SystemExit,
    ):
        pass

    finally:
        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
