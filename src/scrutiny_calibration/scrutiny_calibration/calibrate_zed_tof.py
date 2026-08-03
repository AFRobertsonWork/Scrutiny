#!/usr/bin/env python3

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

OUTPUT_FILE = DATASET_DIR / "calibration.yaml"

PATTERN_SIZE = (9, 6)
SQUARE_SIZE_M = 0.020

MIN_TOF_DEPTH_M = 0.15
MAX_TOF_DEPTH_M = 3.00

MIN_PLANE_POINTS = 250


def load_first_yaml_document(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"YAML file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        documents = list(
            yaml.safe_load_all(file)
        )

    message = next(
        (
            document
            for document in documents
            if isinstance(document, dict)
        ),
        None,
    )

    if message is None:
        raise ValueError(
            f"No valid YAML document found in {path}"
        )

    return message


def load_camera_info(
    path: Path,
) -> dict[str, Any]:
    message = load_first_yaml_document(
        path
    )

    camera_matrix = np.asarray(
        message["k"],
        dtype=np.float64,
    ).reshape(3, 3)

    distortion = np.asarray(
        message["d"],
        dtype=np.float64,
    ).reshape(-1)

    header = message.get(
        "header",
        {},
    )

    if not isinstance(header, dict):
        header = {}

    return {
        "width": int(message["width"]),
        "height": int(message["height"]),
        "K": camera_matrix,
        "D": distortion,
        "distortion_model": str(
            message.get(
                "distortion_model",
                "",
            )
        ),
        "frame_id": str(
            header.get(
                "frame_id",
                "",
            )
        ),
    }


def create_checkerboard_points() -> np.ndarray:
    columns, rows = PATTERN_SIZE

    points = np.zeros(
        (
            columns * rows,
            3,
        ),
        dtype=np.float64,
    )

    grid = np.mgrid[
        0:columns,
        0:rows,
    ].T.reshape(-1, 2)

    points[:, :2] = (
        grid.astype(np.float64)
        * SQUARE_SIZE_M
    )

    return points


def estimate_zed_board_plane(
    corners: np.ndarray,
    zed_info: dict[str, Any],
) -> tuple[
    np.ndarray,
    float,
    float,
]:
    object_points = (
        create_checkerboard_points()
    )

    image_points = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(-1, 2)

    success, rotation_vector, translation_vector = (
        cv2.solvePnP(
            object_points,
            image_points,
            zed_info["K"],
            zed_info["D"],
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    )

    if not success:
        raise RuntimeError(
            "solvePnP failed for the ZED checkerboard."
        )

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    normal = rotation_matrix[:, 2]

    normal /= np.linalg.norm(
        normal
    )

    origin = translation_vector.reshape(3)

    distance = float(
        np.dot(
            normal,
            origin,
        )
    )

    if distance < 0.0:
        normal = -normal
        distance = -distance

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        zed_info["K"],
        zed_info["D"],
    )

    projected = projected.reshape(
        -1,
        2,
    )

    reprojection_error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        projected
                        - image_points
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )

    return (
        normal,
        distance,
        reprojection_error,
    )


def polygon_mask(
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

    kernel = np.ones(
        (9, 9),
        dtype=np.uint8,
    )

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    return mask


def tof_pixels_to_points(
    depth_mm: np.ndarray,
    polygon: np.ndarray,
    tof_info: dict[str, Any],
) -> np.ndarray:
    mask = polygon_mask(
        depth_mm.shape,
        polygon,
    )

    valid = (
        (mask > 0)
        & np.isfinite(depth_mm)
        & (depth_mm > 0)
    )

    rows, columns = np.where(
        valid
    )

    if rows.size < MIN_PLANE_POINTS:
        raise RuntimeError(
            f"Only {rows.size} valid ToF pixels "
            "were found inside the selected board polygon."
        )

    depth_m = (
        depth_mm[
            rows,
            columns,
        ].astype(np.float64)
        / 1000.0
    )

    valid_range = (
        (depth_m >= MIN_TOF_DEPTH_M)
        & (depth_m <= MAX_TOF_DEPTH_M)
    )

    rows = rows[valid_range]
    columns = columns[valid_range]
    depth_m = depth_m[valid_range]

    if depth_m.size < MIN_PLANE_POINTS:
        raise RuntimeError(
            f"Only {depth_m.size} ToF points "
            "remain after depth-range filtering."
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

    x = (
        normalized[:, 0]
        * depth_m
    )

    y = (
        normalized[:, 1]
        * depth_m
    )

    z = depth_m

    points = np.column_stack(
        (
            x,
            y,
            z,
        )
    )

    return points


def fit_plane_svd(
    points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    centroid = np.mean(
        points,
        axis=0,
    )

    centred = (
        points
        - centroid
    )

    _, _, vt = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    normal = vt[-1]

    normal /= np.linalg.norm(
        normal
    )

    return (
        normal,
        centroid,
    )


def fit_plane_robust(
    points: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    int,
    float,
]:
    """
    Robust plane fitting using PCA with iterative
    outlier rejection.

    Returns:
        normal
        plane distance from camera origin
        retained point count
        RMS plane residual in metres
    """

    working = np.asarray(
        points,
        dtype=np.float64,
    ).copy()

    if working.ndim != 2 or working.shape[1] != 3:
        raise ValueError(
            "Plane points must have shape (N, 3)."
        )

    if working.shape[0] < MIN_PLANE_POINTS:
        raise RuntimeError(
            f"Only {working.shape[0]} points were supplied "
            "to the plane fitter."
        )

    for _ in range(6):
        normal, centroid = fit_plane_svd(
            working
        )

        centred = (
            working
            - centroid
        )

        distances = np.abs(
            centred
            @ normal
        )

        median_distance = float(
            np.median(
                distances
            )
        )

        mad = float(
            np.median(
                np.abs(
                    distances
                    - median_distance
                )
            )
        )

        robust_sigma = (
            1.4826
            * mad
        )

        threshold = max(
            0.004,
            median_distance
            + 3.0
            * max(
                robust_sigma,
                0.001,
            ),
        )

        keep = (
            distances
            <= threshold
        )

        retained_count = int(
            np.count_nonzero(
                keep
            )
        )

        if retained_count < MIN_PLANE_POINTS:
            break

        if retained_count == working.shape[0]:
            break

        working = working[
            keep
        ]

    if working.shape[0] < MIN_PLANE_POINTS:
        raise RuntimeError(
            "Not enough ToF plane inliers remained "
            "after outlier rejection."
        )

    normal, centroid = fit_plane_svd(
        working
    )

    centred = (
        working
        - centroid
    )

    distances = np.abs(
        centred
        @ normal
    )

    rms_error = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    distance = float(
        np.dot(
            normal,
            centroid,
        )
    )

    if distance < 0.0:
        normal = -normal
        distance = -distance

    return (
        normal,
        distance,
        int(working.shape[0]),
        rms_error,
    )


def estimate_rotation(
    tof_normals: np.ndarray,
    zed_normals: np.ndarray,
) -> np.ndarray:
    """
    Robustly estimate the rotation mapping ToF
    plane normals into the ZED coordinate frame.
    """

    tof = np.asarray(
        tof_normals,
        dtype=np.float64,
    ).copy()

    zed = np.asarray(
        zed_normals,
        dtype=np.float64,
    ).copy()

    tof /= np.linalg.norm(
        tof,
        axis=1,
        keepdims=True,
    )

    zed /= np.linalg.norm(
        zed,
        axis=1,
        keepdims=True,
    )

    active = np.ones(
        tof.shape[0],
        dtype=bool,
    )

    rotation = np.eye(
        3,
        dtype=np.float64,
    )

    for _ in range(5):
        selected_tof = tof[
            active
        ]

        selected_zed = zed[
            active
        ]

        if selected_tof.shape[0] < 6:
            break

        cross_covariance = (
            selected_zed.T
            @ selected_tof
        )

        u, _, vt = np.linalg.svd(
            cross_covariance
        )

        rotation = (
            u
            @ vt
        )

        if np.linalg.det(
            rotation
        ) < 0.0:
            u[:, -1] *= -1.0

            rotation = (
                u
                @ vt
            )

        predicted = (
            rotation
            @ tof.T
        ).T

        dots = np.sum(
            predicted
            * zed,
            axis=1,
        )

        dots = np.clip(
            dots,
            -1.0,
            1.0,
        )

        errors_deg = np.degrees(
            np.arccos(
                dots
            )
        )

        active_errors = errors_deg[
            active
        ]

        median_error = float(
            np.median(
                active_errors
            )
        )

        mad = float(
            np.median(
                np.abs(
                    active_errors
                    - median_error
                )
            )
        )

        threshold = max(
            2.0,
            median_error
            + 2.5
            * max(
                mad,
                0.25,
            ),
        )

        updated_active = (
            errors_deg
            <= threshold
        )

        if np.array_equal(
            updated_active,
            active,
        ):
            break

        active = updated_active

    print(
        "Rotation samples retained:",
        int(
            np.count_nonzero(
                active
            )
        ),
        "/",
        int(
            active.size
        ),
    )

    return rotation


def estimate_translation(
    rotation: np.ndarray,
    tof_normals: np.ndarray,
    tof_distances: np.ndarray,
    zed_normals: np.ndarray,
    zed_distances: np.ndarray,
) -> np.ndarray:
    rotated_tof_normals = (
        rotation
        @ tof_normals.T
    ).T

    aligned_normals = (
        zed_normals
        + rotated_tof_normals
    )

    normal_lengths = np.linalg.norm(
        aligned_normals,
        axis=1,
        keepdims=True,
    )

    valid = (
        normal_lengths[:, 0]
        > 1e-9
    )

    aligned_normals = (
        aligned_normals[
            valid
        ]
        / normal_lengths[
            valid
        ]
    )

    right_hand_side = (
        zed_distances[
            valid
        ]
        - tof_distances[
            valid
        ]
    )

    translation, _, _, _ = np.linalg.lstsq(
        aligned_normals,
        right_hand_side,
        rcond=None,
    )

    return translation.reshape(3)


def rotation_matrix_to_rpy(
    rotation: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    sy = np.sqrt(
        rotation[0, 0] ** 2
        + rotation[1, 0] ** 2
    )

    singular = (
        sy
        < 1e-9
    )

    if not singular:
        roll = np.arctan2(
            rotation[2, 1],
            rotation[2, 2],
        )

        pitch = np.arctan2(
            -rotation[2, 0],
            sy,
        )

        yaw = np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )
    else:
        roll = np.arctan2(
            -rotation[1, 2],
            rotation[1, 1],
        )

        pitch = np.arctan2(
            -rotation[2, 0],
            sy,
        )

        yaw = 0.0

    return (
        float(roll),
        float(pitch),
        float(yaw),
    )


def save_calibration(
    rotation: np.ndarray,
    translation: np.ndarray,
    used_samples: list[str],
    normal_error_deg: float,
    plane_distance_rms_m: float,
) -> None:
    roll, pitch, yaw = (
        rotation_matrix_to_rpy(
            rotation
        )
    )

    output = {
        "source_frame": (
            "tof_depth_frame"
        ),
        "target_frame": (
            "zed_left_camera_frame_optical"
        ),
        "transform_direction": (
            "tof_to_zed"
        ),
        "rotation_matrix": (
            rotation.reshape(-1).tolist()
        ),
        "translation_m": (
            translation.tolist()
        ),
        "rpy_rad": [
            roll,
            pitch,
            yaw,
        ],
        "rpy_deg": [
            float(
                np.degrees(
                    roll
                )
            ),
            float(
                np.degrees(
                    pitch
                )
            ),
            float(
                np.degrees(
                    yaw
                )
            ),
        ],
        "quality": {
            "sample_count": (
                len(
                    used_samples
                )
            ),
            "mean_normal_error_deg": (
                float(
                    normal_error_deg
                )
            ),
            "plane_distance_rms_m": (
                float(
                    plane_distance_rms_m
                )
            ),
        },
        "used_samples": (
            used_samples
        ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            output,
            file,
            sort_keys=False,
        )


def main() -> None:
    print(
        "\nZED–ToF extrinsic calibration"
    )

    print(
        "Dataset:",
        DATASET_DIR,
    )

    zed_info = load_camera_info(
        ZED_CAMERA_INFO_FILE
    )

    tof_info = load_camera_info(
        TOF_CAMERA_INFO_FILE
    )

    sample_folders = sorted(
        folder
        for folder in DATASET_DIR.glob(
            "sample_*"
        )
        if folder.is_dir()
    )

    tof_normals: list[
        np.ndarray
    ] = []

    tof_distances: list[
        float
    ] = []

    zed_normals: list[
        np.ndarray
    ] = []

    zed_distances: list[
        float
    ] = []

    used_samples: list[
        str
    ] = []

    print(
        f"\nFound {len(sample_folders)} "
        "sample folders.\n"
    )

    for index, folder in enumerate(
        sample_folders,
        start=1,
    ):
        try:
            corners = np.load(
                folder
                / "zed_checkerboard_corners.npy"
            )

            depth_mm = np.load(
                folder
                / "tof_depth.npy"
            )

            polygon = np.load(
                folder
                / "tof_board_polygon.npy"
            )

            (
                zed_normal,
                zed_distance,
                reprojection_error,
            ) = estimate_zed_board_plane(
                corners,
                zed_info,
            )

            tof_points = tof_pixels_to_points(
                depth_mm,
                polygon,
                tof_info,
            )

            (
                tof_normal,
                tof_distance,
                inlier_count,
                tof_plane_rms,
            ) = fit_plane_robust(
                tof_points
            )

            if zed_normal[2] < 0.0:
                zed_normal = (
                    -zed_normal
                )

                zed_distance = (
                    -zed_distance
                )

            if tof_normal[2] < 0.0:
                tof_normal = (
                    -tof_normal
                )

                tof_distance = (
                    -tof_distance
                )

            zed_normals.append(
                zed_normal
            )

            zed_distances.append(
                float(
                    zed_distance
                )
            )

            tof_normals.append(
                tof_normal
            )

            tof_distances.append(
                float(
                    tof_distance
                )
            )

            used_samples.append(
                folder.name
            )

            print(
                f"{index:02d} "
                f"{folder.name}: "
                f"PnP={reprojection_error:.3f}px, "
                f"ToF plane="
                f"{tof_plane_rms * 1000.0:.2f}mm, "
                f"inliers={inlier_count}"
            )

        except Exception as error:
            print(
                f"{index:02d} "
                f"{folder.name}: "
                f"SKIPPED — {error}"
            )

    if len(
        used_samples
    ) < 6:
        raise RuntimeError(
            "Fewer than six usable calibration poses."
        )

    tof_normals_array = np.asarray(
        tof_normals,
        dtype=np.float64,
    )

    zed_normals_array = np.asarray(
        zed_normals,
        dtype=np.float64,
    )

    tof_distances_array = np.asarray(
        tof_distances,
        dtype=np.float64,
    ).reshape(-1)

    zed_distances_array = np.asarray(
        zed_distances,
        dtype=np.float64,
    ).reshape(-1)

    rotation = estimate_rotation(
        tof_normals_array,
        zed_normals_array,
    )

    translation = estimate_translation(
        rotation,
        tof_normals_array,
        tof_distances_array,
        zed_normals_array,
        zed_distances_array,
    )

    predicted_normals = (
        rotation
        @ tof_normals_array.T
    ).T

    normal_dots = np.sum(
        predicted_normals
        * zed_normals_array,
        axis=1,
    )

    normal_dots = np.clip(
        normal_dots,
        -1.0,
        1.0,
    )

    normal_errors_deg = np.degrees(
        np.arccos(
            normal_dots
        )
    )

    predicted_distances = (
        tof_distances_array
        + zed_normals_array
        @ translation
    )

    distance_errors = (
        predicted_distances
        - zed_distances_array
    )

    normal_error_mean = float(
        np.mean(
            normal_errors_deg
        )
    )

    distance_rms = float(
        np.sqrt(
            np.mean(
                distance_errors ** 2
            )
        )
    )

    print(
        "\nCalibration result"
    )

    print(
        "\nRotation matrix (ToF → ZED):"
    )

    print(
        rotation
    )

    print(
        "\nTranslation metres (ToF → ZED):"
    )

    print(
        translation
    )

    print(
        "\nTranslation millimetres:"
    )

    print(
        translation
        * 1000.0
    )

    print(
        "\nMean normal alignment error: "
        f"{normal_error_mean:.3f} degrees"
    )

    print(
        "Plane-distance RMS error: "
        f"{distance_rms * 1000.0:.3f} mm"
    )

    print(
        "Usable samples:",
        len(
            used_samples
        ),
    )

    save_calibration(
        rotation,
        translation,
        used_samples,
        normal_error_mean,
        distance_rms,
    )

    print(
        "\nSaved calibration:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
