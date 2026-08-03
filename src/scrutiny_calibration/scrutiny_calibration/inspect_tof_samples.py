#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path.home() / "scrutiny_calibration_data"
OUTPUT_DIR = DATASET_DIR / "tof_diagnostics"


def colourise_depth(depth_mm: np.ndarray) -> np.ndarray:
    depth = depth_mm.astype(np.float32)

    valid = (
        np.isfinite(depth)
        & (depth > 100.0)
        & (depth < 3000.0)
    )

    output = np.zeros(
        (*depth.shape, 3),
        dtype=np.uint8,
    )

    if np.count_nonzero(valid) < 100:
        return output

    valid_values = depth[valid]

    lower = float(
        np.percentile(valid_values, 2.0)
    )

    upper = float(
        np.percentile(valid_values, 98.0)
    )

    if upper <= lower:
        return output

    normalised = np.zeros_like(
        depth,
        dtype=np.uint8,
    )

    scaled = (
        (depth - lower)
        / (upper - lower)
        * 255.0
    )

    normalised[valid] = np.clip(
        scaled[valid],
        0,
        255,
    ).astype(np.uint8)

    coloured = cv2.applyColorMap(
        255 - normalised,
        cv2.COLORMAP_JET,
    )

    coloured[~valid] = 0

    return coloured


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_folders = sorted(
        folder
        for folder in DATASET_DIR.glob("sample_*")
        if folder.is_dir()
    )

    if not sample_folders:
        raise RuntimeError(
            f"No sample folders found in {DATASET_DIR}"
        )

    print(
        f"Found {len(sample_folders)} samples."
    )

    for index, folder in enumerate(
        sample_folders,
        start=1,
    ):
        depth_file = folder / "tof_depth.npy"

        if not depth_file.exists():
            print(
                f"Skipping {folder.name}: "
                "tof_depth.npy missing"
            )
            continue

        depth = np.load(depth_file)

        visual = colourise_depth(depth)

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        if np.any(valid):
            valid_depth = depth[valid]

            median_mm = float(
                np.median(valid_depth)
            )

            text = (
                f"{folder.name} | "
                f"median {median_mm:.0f} mm"
            )
        else:
            text = (
                f"{folder.name} | no valid depth"
            )

        cv2.putText(
            visual,
            text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        output_file = (
            OUTPUT_DIR
            / f"{index:02d}_{folder.name}.png"
        )

        cv2.imwrite(
            str(output_file),
            visual,
        )

        print(
            f"Saved {output_file.name}"
        )

    print(
        f"\nDiagnostics saved in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
