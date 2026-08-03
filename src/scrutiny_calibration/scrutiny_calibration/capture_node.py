#!/usr/bin/env python3

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


FrameRecord = Tuple[int, int, np.ndarray]


class CalibrationCaptureNode(Node):

    def __init__(self) -> None:
        super().__init__("zed_tof_calibration_capture")

        self.declare_parameter(
            "zed_rgb_topic",
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
            "output_directory",
            str(
                Path.home()
                / "scrutiny_calibration_data_synced"
            ),
        )

        self.declare_parameter(
            "sync_tolerance_ms",
            500.0,
        )

        self.rgb_topic = str(
            self.get_parameter(
                "zed_rgb_topic"
            ).value
        )

        self.zed_depth_topic = str(
            self.get_parameter(
                "zed_depth_topic"
            ).value
        )

        self.tof_depth_topic = str(
            self.get_parameter(
                "tof_depth_topic"
            ).value
        )

        self.output_directory = Path(
            self.get_parameter(
                "output_directory"
            ).value
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sync_tolerance_ns = int(
            float(
                self.get_parameter(
                    "sync_tolerance_ms"
                ).value
            )
            * 1_000_000.0
        )

        self.bridge = CvBridge()

        self.pattern_size = (9, 6)
        self.square_size_mm = 20.0

        self.rgb_buffer: Deque[FrameRecord] = deque(
            maxlen=60
        )

        self.zed_depth_buffer: Deque[FrameRecord] = deque(
            maxlen=60
        )

        self.tof_depth_buffer: Deque[FrameRecord] = deque(
            maxlen=60
        )

        self.latest_rgb: Optional[np.ndarray] = None

        self.synced_rgb: Optional[np.ndarray] = None
        self.synced_zed_depth: Optional[np.ndarray] = None
        self.synced_tof_depth: Optional[np.ndarray] = None

        self.synced_rgb_arrival_ns: Optional[int] = None
        self.synced_zed_arrival_ns: Optional[int] = None
        self.synced_tof_arrival_ns: Optional[int] = None

        self.synced_rgb_header_ns: Optional[int] = None
        self.synced_zed_header_ns: Optional[int] = None
        self.synced_tof_header_ns: Optional[int] = None

        self.latest_corners: Optional[np.ndarray] = None
        self.checkerboard_found = False

        self.last_saved_rgb_arrival_ns: Optional[int] = None

        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        self.zed_depth_subscription = (
            self.create_subscription(
                Image,
                self.zed_depth_topic,
                self.zed_depth_callback,
                qos_profile_sensor_data,
            )
        )

        self.tof_depth_subscription = (
            self.create_subscription(
                Image,
                self.tof_depth_topic,
                self.tof_depth_callback,
                qos_profile_sensor_data,
            )
        )

        self.display_timer = self.create_timer(
            0.05,
            self.display,
        )

        self.get_logger().info(
            "Synchronized calibration capture started"
        )

        self.get_logger().info(
            f"RGB: {self.rgb_topic}"
        )

        self.get_logger().info(
            f"ZED depth: {self.zed_depth_topic}"
        )

        self.get_logger().info(
            f"ToF depth: {self.tof_depth_topic}"
        )

        self.get_logger().info(
            "Synchronization tolerance: "
            f"{self.sync_tolerance_ns / 1_000_000.0:.1f} ms"
        )

        self.get_logger().info(
            f"Output: {self.output_directory}"
        )

        self.get_logger().info(
            "Press S to save. Press Q to quit."
        )

    @staticmethod
    def header_stamp_ns(
        message: Image,
    ) -> int:
        return (
            int(message.header.stamp.sec)
            * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def arrival_stamp_ns(self) -> int:
        return int(
            self.get_clock().now().nanoseconds
        )

    def rgb_callback(
        self,
        message: Image,
    ) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

        except Exception as error:
            self.get_logger().error(
                f"RGB conversion failed: {error}"
            )
            return

        arrival_ns = self.arrival_stamp_ns()
        header_ns = self.header_stamp_ns(
            message
        )

        frame = np.asarray(
            frame
        ).copy()

        self.latest_rgb = frame.copy()

        self.rgb_buffer.append(
            (
                arrival_ns,
                header_ns,
                frame,
            )
        )

        self.try_create_synchronized_bundle(
            arrival_ns
        )

    def zed_depth_callback(
        self,
        message: Image,
    ) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )

        except Exception as error:
            self.get_logger().error(
                f"ZED depth conversion failed: {error}"
            )
            return

        self.zed_depth_buffer.append(
            (
                self.arrival_stamp_ns(),
                self.header_stamp_ns(
                    message
                ),
                np.asarray(
                    frame
                ).copy(),
            )
        )

        if self.rgb_buffer:
            self.try_create_synchronized_bundle(
                self.rgb_buffer[-1][0]
            )

    def tof_depth_callback(
        self,
        message: Image,
    ) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )

        except Exception as error:
            self.get_logger().error(
                f"ToF depth conversion failed: {error}"
            )
            return

        self.tof_depth_buffer.append(
            (
                self.arrival_stamp_ns(),
                self.header_stamp_ns(
                    message
                ),
                np.asarray(
                    frame
                ).copy(),
            )
        )

        if self.rgb_buffer:
            self.try_create_synchronized_bundle(
                self.rgb_buffer[-1][0]
            )

    @staticmethod
    def nearest_frame(
        buffer: Deque[FrameRecord],
        target_arrival_ns: int,
    ) -> Optional[FrameRecord]:
        if not buffer:
            return None

        return min(
            buffer,
            key=lambda record: abs(
                record[0]
                - target_arrival_ns
            ),
        )

    def try_create_synchronized_bundle(
        self,
        rgb_arrival_ns: int,
    ) -> None:
        rgb_record = self.nearest_frame(
            self.rgb_buffer,
            rgb_arrival_ns,
        )

        zed_record = self.nearest_frame(
            self.zed_depth_buffer,
            rgb_arrival_ns,
        )

        tof_record = self.nearest_frame(
            self.tof_depth_buffer,
            rgb_arrival_ns,
        )

        if (
            rgb_record is None
            or zed_record is None
            or tof_record is None
        ):
            return

        zed_difference_ns = abs(
            zed_record[0]
            - rgb_record[0]
        )

        tof_difference_ns = abs(
            tof_record[0]
            - rgb_record[0]
        )

        if (
            zed_difference_ns
            > self.sync_tolerance_ns
        ):
            return

        if (
            tof_difference_ns
            > self.sync_tolerance_ns
        ):
            return

        self.synced_rgb_arrival_ns = (
            rgb_record[0]
        )

        self.synced_zed_arrival_ns = (
            zed_record[0]
        )

        self.synced_tof_arrival_ns = (
            tof_record[0]
        )

        self.synced_rgb_header_ns = (
            rgb_record[1]
        )

        self.synced_zed_header_ns = (
            zed_record[1]
        )

        self.synced_tof_header_ns = (
            tof_record[1]
        )

        self.synced_rgb = (
            rgb_record[2].copy()
        )

        self.synced_zed_depth = (
            zed_record[2].copy()
        )

        self.synced_tof_depth = (
            tof_record[2].copy()
        )

    def detect_checkerboard(
        self,
        image: np.ndarray,
    ) -> tuple[
        bool,
        Optional[np.ndarray],
    ]:
        grey = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        found, corners = cv2.findChessboardCorners(
            grey,
            self.pattern_size,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            ),
        )

        if (
            not found
            or corners is None
        ):
            return False, None

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners = cv2.cornerSubPix(
            grey,
            corners,
            (5, 5),
            (-1, -1),
            criteria,
        )

        return (
            True,
            corners.astype(
                np.float32
            ),
        )

    def save_sample(self) -> None:
        if (
            self.synced_rgb is None
            or self.synced_zed_depth is None
            or self.synced_tof_depth is None
        ):
            self.get_logger().warning(
                "No synchronized bundle available"
            )
            return

        if (
            not self.checkerboard_found
            or self.latest_corners is None
        ):
            self.get_logger().warning(
                "Checkerboard not detected"
            )
            return

        if (
            self.synced_rgb_arrival_ns is None
            or self.synced_zed_arrival_ns is None
            or self.synced_tof_arrival_ns is None
        ):
            self.get_logger().warning(
                "Synchronized timestamps unavailable"
            )
            return

        if (
            self.last_saved_rgb_arrival_ns
            == self.synced_rgb_arrival_ns
        ):
            self.get_logger().warning(
                "This synchronized bundle was already saved"
            )
            return

        zed_delta_ms = (
            self.synced_zed_arrival_ns
            - self.synced_rgb_arrival_ns
        ) / 1_000_000.0

        tof_delta_ms = (
            self.synced_tof_arrival_ns
            - self.synced_rgb_arrival_ns
        ) / 1_000_000.0

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        sample_directory = (
            self.output_directory
            / f"sample_{timestamp}"
        )

        sample_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        cv2.imwrite(
            str(
                sample_directory
                / "zed_rgb.png"
            ),
            self.synced_rgb,
        )

        np.save(
            sample_directory
            / "zed_depth.npy",
            self.synced_zed_depth,
        )

        np.save(
            sample_directory
            / "tof_depth.npy",
            self.synced_tof_depth,
        )

        np.save(
            sample_directory
            / "zed_checkerboard_corners.npy",
            self.latest_corners,
        )

        metadata = (
            "capture_type: arrival_time_synchronized\n"
            "pattern_inner_corners: 9x6\n"
            f"square_size_mm: {self.square_size_mm}\n"
            f"rgb_topic: {self.rgb_topic}\n"
            f"zed_depth_topic: {self.zed_depth_topic}\n"
            f"tof_depth_topic: {self.tof_depth_topic}\n"
            f"rgb_arrival_ns: {self.synced_rgb_arrival_ns}\n"
            f"zed_arrival_ns: {self.synced_zed_arrival_ns}\n"
            f"tof_arrival_ns: {self.synced_tof_arrival_ns}\n"
            f"zed_minus_rgb_ms: {zed_delta_ms:.3f}\n"
            f"tof_minus_rgb_ms: {tof_delta_ms:.3f}\n"
            f"rgb_header_ns: {self.synced_rgb_header_ns}\n"
            f"zed_header_ns: {self.synced_zed_header_ns}\n"
            f"tof_header_ns: {self.synced_tof_header_ns}\n"
            f"rgb_shape: {self.synced_rgb.shape}\n"
            f"zed_depth_shape: {self.synced_zed_depth.shape}\n"
            f"tof_depth_shape: {self.synced_tof_depth.shape}\n"
        )

        with (
            sample_directory
            / "metadata.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            metadata_file.write(
                metadata
            )

        self.last_saved_rgb_arrival_ns = (
            self.synced_rgb_arrival_ns
        )

        self.get_logger().info(
            f"Saved: {sample_directory.name}"
        )

        self.get_logger().info(
            f"ZED delta: {zed_delta_ms:+.1f} ms | "
            f"ToF delta: {tof_delta_ms:+.1f} ms"
        )

    def display(self) -> None:
        if self.latest_rgb is None:
            return

        display_image = (
            self.latest_rgb.copy()
        )

        bundle_available = (
            self.synced_rgb is not None
            and self.synced_zed_depth is not None
            and self.synced_tof_depth is not None
        )

        found, corners = (
            self.detect_checkerboard(
                display_image
            )
        )

        self.checkerboard_found = found
        self.latest_corners = corners

        if (
            found
            and corners is not None
        ):
            cv2.drawChessboardCorners(
                display_image,
                self.pattern_size,
                corners,
                found,
            )

        if (
            bundle_available
            and found
        ):
            status = (
                "SYNCED + CHECKERBOARD DETECTED - PRESS S"
            )

            colour = (
                0,
                255,
                0,
            )

        elif bundle_available:
            status = (
                "SYNCED - CHECKERBOARD NOT DETECTED"
            )

            colour = (
                0,
                165,
                255,
            )

        else:
            status = (
                "RGB RECEIVED - WAITING FOR DEPTH SYNC"
            )

            colour = (
                0,
                0,
                255,
            )

        cv2.putText(
            display_image,
            status,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
            cv2.LINE_AA,
        )

        if (
            bundle_available
            and self.synced_rgb_arrival_ns is not None
            and self.synced_zed_arrival_ns is not None
            and self.synced_tof_arrival_ns is not None
        ):
            zed_delta_ms = (
                self.synced_zed_arrival_ns
                - self.synced_rgb_arrival_ns
            ) / 1_000_000.0

            tof_delta_ms = (
                self.synced_tof_arrival_ns
                - self.synced_rgb_arrival_ns
            ) / 1_000_000.0

            cv2.putText(
                display_image,
                (
                    f"ZED: {zed_delta_ms:+.1f} ms | "
                    f"ToF: {tof_delta_ms:+.1f} ms"
                ),
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            "ZED-ToF Synchronized Calibration Capture",
            display_image,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("s"),
            ord("S"),
        ):
            self.save_sample()

        elif key in (
            ord("q"),
            ord("Q"),
        ):
            raise SystemExit

    def destroy_node(self) -> None:
        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(
        args=args
    )

    node = CalibrationCaptureNode()

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
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
