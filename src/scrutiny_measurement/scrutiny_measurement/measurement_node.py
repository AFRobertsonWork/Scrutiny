#!/usr/bin/env python3


import math
from typing import Optional


import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String




class MeasurementNode(Node):
    """Measure a centred object's width using ZED RGB and registered depth."""


    def __init__(self) -> None:
        super().__init__("scrutiny_measurement_node")


        # Topics
        self.declare_parameter(
            "rgb_topic",
            "/zed/zed_node/left/color/rect/image",
        )
        self.declare_parameter(
            "depth_topic",
            "/zed/zed_node/depth/depth_registered",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/zed/zed_node/left/color/rect/camera_info",
        )
        self.declare_parameter(
            "diameter_topic",
            "/scrutiny/object_diameter",
        )
        self.declare_parameter(
            "annotated_topic",
            "/scrutiny/measurement/image",
        )


        # Object detection
        self.declare_parameter("minimum_contour_area", 2500.0)
        self.declare_parameter("maximum_contour_area_ratio", 0.65)


        # Operator stencil
        self.declare_parameter("stencil_width_ratio", 0.55)
        self.declare_parameter("stencil_height_ratio", 0.70)


        # Valid depth range
        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 3.0)


        # Depth range around the centre point used for segmentation
        self.declare_parameter("depth_band_m", 0.08)


        # Search area above and below the object's centroid
        self.declare_parameter("scan_region_ratio", 0.55)


        # Percentage of widest rows used for the final stable width
        self.declare_parameter("widest_row_fraction", 0.30)


        # Leave at 1.0 until validation shows a consistent scale error
        self.declare_parameter("calibration_scale", 1.0)


        self.rgb_topic = str(
            self.get_parameter("rgb_topic").value
        )
        self.depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )


        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )


        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )


        self.bridge = CvBridge()


        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.fx: Optional[float] = None


        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            image_qos,
        )


        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            image_qos,
        )


        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            camera_info_qos,
        )


        self.diameter_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("diameter_topic").value),
            10,
        )


        self.status_publisher = self.create_publisher(
            String,
            "/scrutiny/measurement_status",
            10,
        )


        self.annotated_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("annotated_topic").value),
            image_qos,
        )


        self.timer = self.create_timer(0.1, self.process_frame)


        self.get_logger().info("Scrutiny measurement node started")
        self.get_logger().info(f"RGB: {self.rgb_topic}")
        self.get_logger().info(f"Depth: {self.depth_topic}")
        self.get_logger().info(
            f"Camera info: {self.camera_info_topic}"
        )


    def rgb_callback(self, message: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(
                f"RGB conversion failed: {exc}"
            )


    def depth_callback(self, message: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )


            depth = np.asarray(depth, dtype=np.float32)


            if message.encoding in ("16UC1", "mono16"):
                depth *= 0.001


            self.latest_depth = depth


        except Exception as exc:
            self.get_logger().error(
                f"Depth conversion failed: {exc}"
            )


    def camera_info_callback(self, message: CameraInfo) -> None:
        if len(message.k) < 1 or message.k[0] <= 0.0:
            return


        if self.fx is None:
            self.get_logger().info(
                f"Camera calibration received: fx={message.k[0]:.2f}"
            )


        self.fx = float(message.k[0])


    def process_frame(self) -> None:
        if self.latest_rgb is None:
            self.publish_status("WAITING_FOR_RGB")
            return


        frame = self.latest_rgb.copy()
        annotated = frame.copy()


        height, width = frame.shape[:2]


        stencil_mask = self.create_stencil_mask(
            width,
            height,
        )


        self.draw_stencil(annotated)


        contour = self.detect_object(
            frame,
            stencil_mask,
        )


        if contour is None:
            self.draw_message(
                annotated,
                "Align object inside stencil",
                (30, 45),
                (0, 0, 255),
            )
            self.publish_status("NO_OBJECT")
            self.publish_annotated(annotated)
            return


        cv2.drawContours(
            annotated,
            [contour],
            -1,
            (0, 255, 0),
            3,
        )


        measurement = self.measure_stable_width(
            contour,
            width,
            height,
        )


        if measurement is None:
            self.draw_message(
                annotated,
                "Unable to find stable width",
                (30, 45),
                (0, 165, 255),
            )
            self.publish_status("INVALID_WIDTH")
            self.publish_annotated(annotated)
            return


        diameter_px, left_point, right_point = measurement


        cv2.line(
            annotated,
            left_point,
            right_point,
            (0, 0, 255),
            4,
        )


        cv2.circle(
            annotated,
            left_point,
            6,
            (0, 255, 255),
            thickness=-1,
        )


        cv2.circle(
            annotated,
            right_point,
            6,
            (0, 255, 255),
            thickness=-1,
        )


        depth_m = self.get_contour_depth(
            contour,
            width,
            height,
        )


        if depth_m is None:
            self.draw_message(
                annotated,
                "Object found - waiting for valid depth",
                (30, 45),
                (0, 165, 255),
            )
            self.publish_status("INVALID_DEPTH")
            self.publish_annotated(annotated)
            return


        if self.fx is None or self.fx <= 0.0:
            self.draw_message(
                annotated,
                "Waiting for camera calibration",
                (30, 45),
                (0, 165, 255),
            )
            self.publish_status("WAITING_FOR_CAMERA_INFO")
            self.publish_annotated(annotated)
            return


        calibration_scale = float(
            self.get_parameter("calibration_scale").value
        )


        diameter_mm = (
            diameter_px
            * depth_m
            / self.fx
            * 1000.0
            * calibration_scale
        )


        diameter_message = Float32()
        diameter_message.data = float(diameter_mm)
        self.diameter_publisher.publish(diameter_message)


        self.publish_status("MEASUREMENT_VALID")


        text_x = max(
            10,
            min(left_point[0], right_point[0]),
        )
        text_y = max(
            35,
            left_point[1] - 15,
        )


        cv2.putText(
            annotated,
            f"{diameter_mm:.1f} mm",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )


        cv2.putText(
            annotated,
            f"{diameter_mm:.1f} mm",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


        self.draw_message(
            annotated,
            f"Diameter: {diameter_mm:.1f} mm",
            (30, 45),
            (0, 255, 0),
        )


        self.draw_message(
            annotated,
            f"ZED depth: {depth_m:.3f} m",
            (30, 85),
            (255, 255, 255),
        )


        self.draw_message(
            annotated,
            f"Pixel diameter: {diameter_px:.1f} px",
            (30, 125),
            (255, 255, 255),
        )


        self.publish_annotated(annotated)


    def create_stencil_mask(
        self,
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )


        centre = (
            width // 2,
            height // 2,
        )


        axes = (
            int(
                width
                * float(
                    self.get_parameter(
                        "stencil_width_ratio"
                    ).value
                )
                / 2.0
            ),
            int(
                height
                * float(
                    self.get_parameter(
                        "stencil_height_ratio"
                    ).value
                )
                / 2.0
            ),
        )


        cv2.ellipse(
            mask,
            centre,
            axes,
            0,
            0,
            360,
            255,
            thickness=-1,
        )


        return mask


    def draw_stencil(
        self,
        image: np.ndarray,
    ) -> None:
        height, width = image.shape[:2]


        centre = (
            width // 2,
            height // 2,
        )


        axes = (
            int(
                width
                * float(
                    self.get_parameter(
                        "stencil_width_ratio"
                    ).value
                )
                / 2.0
            ),
            int(
                height
                * float(
                    self.get_parameter(
                        "stencil_height_ratio"
                    ).value
                )
                / 2.0
            ),
        )


        cv2.ellipse(
            image,
            centre,
            axes,
            0,
            0,
            360,
            (0, 255, 255),
            thickness=3,
        )


        cv2.drawMarker(
            image,
            centre,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )


    def detect_object(
        self,
        frame: np.ndarray,
        stencil_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Find the depth-connected object nearest the stencil centre."""


        if self.latest_depth is None:
            return None


        height, width = frame.shape[:2]


        depth = self.latest_depth
        depth_height, depth_width = depth.shape[:2]


        centre_x = width // 2
        centre_y = height // 2


        depth_x = int(
            centre_x * depth_width / float(width)
        )
        depth_y = int(
            centre_y * depth_height / float(height)
        )


        patch_radius = 10


        patch = depth[
            max(0, depth_y - patch_radius):
            min(depth_height, depth_y + patch_radius + 1),
            max(0, depth_x - patch_radius):
            min(depth_width, depth_x + patch_radius + 1),
        ]


        minimum_depth = float(
            self.get_parameter("depth_min_m").value
        )
        maximum_depth = float(
            self.get_parameter("depth_max_m").value
        )


        valid_patch = patch[
            np.isfinite(patch)
            & (patch >= minimum_depth)
            & (patch <= maximum_depth)
        ]


        if valid_patch.size < 20:
            return None


        centre_depth = float(
            np.median(valid_patch)
        )


        depth_band = float(
            self.get_parameter("depth_band_m").value
        )


        depth_mask = (
            np.isfinite(depth)
            & (depth >= centre_depth - depth_band)
            & (depth <= centre_depth + depth_band)
        ).astype(np.uint8) * 255


        depth_mask = cv2.resize(
            depth_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )


        object_mask = cv2.bitwise_and(
            depth_mask,
            stencil_mask,
        )


        kernel = np.ones(
            (7, 7),
            dtype=np.uint8,
        )


        object_mask = cv2.morphologyEx(
            object_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )


        object_mask = cv2.morphologyEx(
            object_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )


        contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )


        minimum_area = float(
            self.get_parameter(
                "minimum_contour_area"
            ).value
        )


        maximum_ratio = float(
            self.get_parameter(
                "maximum_contour_area_ratio"
            ).value
        )


        maximum_area = (
            width * height * maximum_ratio
        )


        valid_contours = [
            contour
            for contour in contours
            if minimum_area
            <= cv2.contourArea(contour)
            <= maximum_area
        ]


        if not valid_contours:
            return None


        centre_point = (
            float(centre_x),
            float(centre_y),
        )


        containing_contours = [
            contour
            for contour in valid_contours
            if cv2.pointPolygonTest(
                contour,
                centre_point,
                False,
            ) >= 0
        ]


        if containing_contours:
            return max(
                containing_contours,
                key=cv2.contourArea,
            )


        def contour_centre_distance(
            candidate: np.ndarray,
        ) -> float:
            moments = cv2.moments(candidate)


            if moments["m00"] == 0:
                return float("inf")


            candidate_x = (
                moments["m10"] / moments["m00"]
            )
            candidate_y = (
                moments["m01"] / moments["m00"]
            )


            return math.hypot(
                candidate_x - centre_x,
                candidate_y - centre_y,
            )


        return min(
            valid_contours,
            key=contour_centre_distance,
        )


    def measure_stable_width(
        self,
        contour: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[
        tuple[
            float,
            tuple[int, int],
            tuple[int, int],
        ]
    ]:
        """
        Measure the widest stable section of a filled object silhouette.


        Several horizontal rows are measured. The final result uses the median
        of the widest group, avoiding both narrow shoulders and isolated noise.
        """


        contour_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )


        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            thickness=-1,
        )


        x, y, bounding_width, bounding_height = cv2.boundingRect(
            contour
        )


        if bounding_width < 2 or bounding_height < 2:
            return None


        scan_region_ratio = float(
            self.get_parameter("scan_region_ratio").value
        )


        scan_height = max(
            1,
            int(bounding_height * scan_region_ratio),
        )


        scan_top = max(
            0,
            y + (bounding_height - scan_height) // 2,
        )


        scan_bottom = min(
            height - 1,
            scan_top + scan_height - 1,
        )


        measurements: list[
            tuple[int, int, int, int]
        ] = []


        for row_y in range(
            scan_top,
            scan_bottom + 1,
        ):
            x_values = np.where(
                contour_mask[row_y] == 255
            )[0]


            if x_values.size < 2:
                continue


            left_x = int(x_values[0])
            right_x = int(x_values[-1])
            row_width = right_x - left_x + 1


            measurements.append(
                (
                    row_width,
                    left_x,
                    right_x,
                    row_y,
                )
            )


        if len(measurements) < 3:
            return None


        measurements.sort(
            key=lambda item: item[0],
            reverse=True,
        )


        widest_fraction = float(
            self.get_parameter(
                "widest_row_fraction"
            ).value
        )


        widest_fraction = min(
            1.0,
            max(0.05, widest_fraction),
        )


        group_size = max(
            3,
            int(
                round(
                    len(measurements)
                    * widest_fraction
                )
            ),
        )


        widest_group = measurements[:group_size]


        group_widths = np.array(
            [
                item[0]
                for item in widest_group
            ],
            dtype=np.float32,
        )


        stable_width = float(
            np.median(group_widths)
        )


        selected = min(
            widest_group,
            key=lambda item: abs(
                item[0] - stable_width
            ),
        )


        diameter_px = float(selected[0])
        left_x = int(selected[1])
        right_x = int(selected[2])
        selected_y = int(selected[3])


        return (
            diameter_px,
            (left_x, selected_y),
            (right_x, selected_y),
        )


    def get_contour_depth(
        self,
        contour: np.ndarray,
        rgb_width: int,
        rgb_height: int,
    ) -> Optional[float]:
        if self.latest_depth is None:
            return None


        depth = self.latest_depth
        depth_height, depth_width = depth.shape[:2]


        scaled_contour = (
            contour.astype(np.float32).copy()
        )


        scaled_contour[:, :, 0] *= (
            depth_width / float(rgb_width)
        )
        scaled_contour[:, :, 1] *= (
            depth_height / float(rgb_height)
        )


        scaled_contour = scaled_contour.astype(
            np.int32
        )


        contour_mask = np.zeros(
            (depth_height, depth_width),
            dtype=np.uint8,
        )


        cv2.drawContours(
            contour_mask,
            [scaled_contour],
            -1,
            255,
            thickness=-1,
        )


        values = depth[
            contour_mask == 255
        ]


        minimum_depth = float(
            self.get_parameter("depth_min_m").value
        )
        maximum_depth = float(
            self.get_parameter("depth_max_m").value
        )


        valid_values = values[
            np.isfinite(values)
            & (values >= minimum_depth)
            & (values <= maximum_depth)
        ]


        if valid_values.size < 50:
            return None


        return float(
            np.median(valid_values)
        )


    def publish_status(
        self,
        status: str,
    ) -> None:
        message = String()
        message.data = status
        self.status_publisher.publish(message)


    def publish_annotated(
        self,
        frame: np.ndarray,
    ) -> None:
        try:
            message = self.bridge.cv2_to_imgmsg(
                frame,
                encoding="bgr8",
            )
            self.annotated_publisher.publish(message)


        except Exception as exc:
            self.get_logger().error(
                f"Annotated image publication failed: {exc}"
            )


    @staticmethod
    def draw_message(
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )


        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
            cv2.LINE_AA,
        )




def main(args=None) -> None:
    rclpy.init(args=args)
    node = MeasurementNode()


    try:
        rclpy.spin(node)


    except KeyboardInterrupt:
        pass


    finally:
        node.destroy_node()


        if rclpy.ok():
            rclpy.shutdown()




if __name__ == "__main__":
    main()