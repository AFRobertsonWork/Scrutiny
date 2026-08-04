#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


IMAGE_TOPICS = {
    "/zed/zed_node/left/color/rect/image": "zed_rgb",
    "/zed/zed_node/depth/depth_registered": "zed_depth",
    "/scrutiny/tof/depth_raw": "tof_depth_raw",
    "/scrutiny/tof/depth8": "tof_depth8",
    "/scrutiny/measurement/fused_overlay": "fused_overlay",
}

DIAMETER_TOPIC = "/scrutiny/measurement/diameter_mm"


def open_reader(bag_path: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def save_depth_png(path: Path, image: np.ndarray):
    depth = np.asarray(image)

    if depth.dtype == np.uint16:
        cv2.imwrite(str(path), depth)
        return

    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    if not np.any(valid):
        cv2.imwrite(str(path), np.zeros(depth.shape, dtype=np.uint16))
        return

    if float(np.median(depth[valid])) < 20.0:
        depth = depth * 1000.0

    depth[~valid] = 0
    depth = np.clip(depth, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-step", type=int, default=1)
    args = parser.parse_args()

    bag_path = args.bag.expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else bag_path.parent / f"{bag_path.name}_export"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in IMAGE_TOPICS.values():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    reader = open_reader(bag_path)
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    msg_classes = {topic: get_message(type_name) for topic, type_name in topic_types.items()}

    bridge = CvBridge()
    topic_counts = {topic: 0 for topic in topic_types}
    seen_counts = {topic: 0 for topic in IMAGE_TOPICS}
    exported_counts = {topic: 0 for topic in IMAGE_TOPICS}
    diameter_rows = []

    first_timestamp_ns = None
    last_timestamp_ns = None

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if first_timestamp_ns is None:
            first_timestamp_ns = timestamp_ns
        last_timestamp_ns = timestamp_ns

        if topic in IMAGE_TOPICS:
            seen_counts[topic] += 1
            if (seen_counts[topic] - 1) % max(args.image_step, 1) != 0:
                continue

            msg = deserialize_message(data, msg_classes[topic])
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            exported_counts[topic] += 1
            filename = (
                output_dir
                / IMAGE_TOPICS[topic]
                / f"{exported_counts[topic]:06d}_{timestamp_ns}.png"
            )

            if topic in {
                "/zed/zed_node/depth/depth_registered",
                "/scrutiny/tof/depth_raw",
            }:
                save_depth_png(filename, image)
            else:
                cv2.imwrite(str(filename), np.asarray(image))

        elif topic == DIAMETER_TOPIC:
            msg = deserialize_message(data, msg_classes[topic])
            relative_time_s = (
                0.0
                if first_timestamp_ns is None
                else (timestamp_ns - first_timestamp_ns) / 1e9
            )
            diameter_rows.append(
                {
                    "timestamp_ns": timestamp_ns,
                    "relative_time_s": relative_time_s,
                    "diameter_mm": float(msg.data),
                }
            )

    with (output_dir / "diameter_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp_ns", "relative_time_s", "diameter_mm"],
        )
        writer.writeheader()
        writer.writerows(diameter_rows)

    duration_s = (
        0.0
        if first_timestamp_ns is None or last_timestamp_ns is None
        else (last_timestamp_ns - first_timestamp_ns) / 1e9
    )

    summary = {
        "bag_path": str(bag_path),
        "duration_s": duration_s,
        "topic_counts": topic_counts,
        "exported_image_counts": exported_counts,
        "diameter_measurement_count": len(diameter_rows),
    }

    with (output_dir / "export_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Export complete: {output_dir}")
    print(f"Diameter rows: {len(diameter_rows)}")
    for topic, count in exported_counts.items():
        print(f"{topic}: {count} images")


if __name__ == "__main__":
    main()
