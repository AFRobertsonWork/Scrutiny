#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DATASET_DIR = Path.home() / "scrutiny_calibration_data_synced"

ZED_CAMERA_INFO_FILE = (
    DATASET_DIR
    / "camera_info"
    / "zed_left_camera_info.yaml"
)

TOF_CAMERA_INFO_FILE = (
    DATASET_DIR
    / "camera_info"
    / "tof_camera_info.yaml"
)

CALIBRATION_FILE = (
    DATASET_DIR
    / "calibration.yaml"
)

OUTPUT_DIR = (
    DATASET_DIR
    / "projection_validation"
)


def load_first_yaml_document(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        documents = list(
            yaml.safe_load_all(file)
        )

    document = next(
        (
            item
            for item in documents
            if isinstance(item, dict)
        ),
        None,
    )

    if document is None:
        raise ValueError(
            f"No valid YAML document found in {path}"
        )

    return document


def load_camera_info(
    path: Path,
) -> dict[str, Any]:
    message = load_first_yaml_document(
        path
    )

    return {
        "width": int(message["width"]),
        "height": int(message["height"]),
        "K": np.asarray(
            message["k"],
            dtype=np.float64,
        ).reshape(3, 3),
        "D": np.asarray(
            message["d"],
            dtype=np.float64,
        ).reshape(-1),
    }


def load_calibration(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    message = load_first_yaml_document(
        path
    )

    rotation = np.asarray(
        message["rotation_matrix"],
        dtype=np.float64,
    ).reshape(3, 3)

    translation = np.asarray(
        message["translation_m"],
        dtype=np.float64,
    ).reshape(3)

    return rotation, translation


def create_polygon_mask(
    shape: tuple[int, int],
    polygon: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    polygon = np.asarray(
        polygon,
        dtype=np.int32,
    ).reshape(-1, 2)

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    # Remove uncertain edge pixels.
    kernel = np.ones(
        (7, 7),
        dtype=np.uint8,
    )

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    return mask


def tof_depth_to_points(
    depth_mm: np.ndarray,
    polygon: np.ndarray,
    tof_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mask = create_polygon_mask(
        depth_mm.shape,
        polygon,
    )

    valid = (
        (mask > 0)
        & np.isfinite(depth_mm)
        & (depth_mm >= 150)
        & (depth_mm <= 3000)
    )

    rows, columns = np.where(
        valid
    )

    if rows.size == 0:
        raise RuntimeError(
            "No valid ToF depth points inside the polygon."
        )

    depth_m = (
        depth_mm[rows, columns]
        .astype(np.float64)
        / 1000.0
    )

    pixels = np.column_stack(
        (
            columns.astype(np.float64),
            rows.astype(np.float64),
        )
    ).reshape(-1, 1, 2)

    normalized = cv2.undistortPoints(
        pixels,
        tof_info["K"],
        tof_info["D"],
    ).reshape(-1, 2)

    points = np.column_stack(
        (
            normalized[:, 0] * depth_m,
            normalized[:, 1] * depth_m,
            depth_m,
        )
    )

    return points, depth_m


def transform_points(
    points_tof: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        rotation
        @ points_tof.T
    ).T + translation


def project_to_zed(
    points_zed: np.ndarray,
    zed_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(points_zed).all(axis=1)
        & (points_zed[:, 2] > 0.05)
    )

    points = points_zed[valid]

    if points.shape[0] == 0:
        raise RuntimeError(
            "No transformed points are in front of the ZED camera."
        )

    rotation_vector = np.zeros(
        (3, 1),
        dtype=np.float64,
    )

    translation_vector = np.zeros(
        (3, 1),
        dtype=np.float64,
    )

    projected, _ = cv2.projectPoints(
        points,
        rotation_vector,
        translation_vector,
        zed_info["K"],
        zed_info["D"],
    )

    return (
        projected.reshape(-1, 2),
        points[:, 2],
    )


def draw_projection(
    rgb: np.ndarray,
    projected: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    output = rgb.copy()

    height, width = output.shape[:2]

    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
        & np.isfinite(projected).all(axis=1)
    )

    projected = projected[inside]
    depths = depths[inside]

    if projected.shape[0] == 0:
        raise RuntimeError(
            "No ToF points project inside the ZED image."
        )

    minimum_depth = float(
        np.percentile(depths, 2)
    )

    maximum_depth = float(
        np.percentile(depths, 98)
    )

    depth_range = max(
        maximum_depth - minimum_depth,
        1e-6,
    )

    normalized = np.clip(
        (
            depths - minimum_depth
        )
        / depth_range
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    colour_strip = cv2.applyColorMap(
        normalized.reshape(-1, 1),
        cv2.COLORMAP_JET,
    ).reshape(-1, 3)

    # Draw every second point to reduce clutter.
    for index in range(
        0,
        projected.shape[0],
        2,
    ):
        x = int(
            round(projected[index, 0])
        )

        y = int(
            round(projected[index, 1])
        )

        colour = tuple(
            int(value)
            for value in colour_strip[index]
        )

        cv2.circle(
            output,
            (x, y),
            1,
            colour,
            thickness=-1,
        )

    cv2.putText(
        output,
        "Projected ToF board points",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help=(
            "One-based sample number. "
            "Default: 15"
        ),
    )

    arguments = parser.parse_args()

    sample_folders = sorted(
        folder
        for folder in DATASET_DIR.glob(
            "sample_*"
        )
        if folder.is_dir()
    )

    if not sample_folders:
        raise RuntimeError(
            "No calibration samples found."
        )

    sample_index = (
        arguments.sample - 1
    )

    if (
        sample_index < 0
        or sample_index >= len(sample_folders)
    ):
        raise ValueError(
            f"Sample must be between 1 and "
            f"{len(sample_folders)}."
        )

    sample_folder = sample_folders[
        sample_index
    ]

    print(
        "Validating:",
        sample_folder.name,
    )

    rgb = cv2.imread(
        str(
            sample_folder
            / "zed_rgb.png"
        ),
        cv2.IMREAD_COLOR,
    )

    if rgb is None:
        raise RuntimeError(
            "Could not load ZED RGB image."
        )

    tof_depth = np.load(
        sample_folder
        / "tof_depth.npy"
    )

    polygon = np.load(
        sample_folder
        / "tof_board_polygon.npy"
    )

    zed_info = load_camera_info(
        ZED_CAMERA_INFO_FILE
    )

    tof_info = load_camera_info(
        TOF_CAMERA_INFO_FILE
    )

    rotation, translation = load_calibration(
        CALIBRATION_FILE
    )

    tof_points, _ = tof_depth_to_points(
        tof_depth,
        polygon,
        tof_info,
    )

    zed_points = transform_points(
        tof_points,
        rotation,
        translation,
    )

    projected, projected_depth = project_to_zed(
        zed_points,
        zed_info,
    )

    output = draw_projection(
        rgb,
        projected,
        projected_depth,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{sample_folder.name}_projection.png"
    )

    cv2.imwrite(
        str(output_file),
        output,
    )

    print(
        "\nSaved projection:"
    )

    print(
        output_file
    )

    cv2.imshow(
        "ZED-ToF projection validation",
        output,
    )

    print(
        "\nPress any key in the image window to close."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()