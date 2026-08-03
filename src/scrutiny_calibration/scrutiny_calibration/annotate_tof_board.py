#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path.home() / "scrutiny_calibration_data_synced"
ANNOTATION_FILENAME = "tof_board_polygon.npy"
OVERLAY_FILENAME = "tof_board_polygon.png"

WINDOW_NAME = "Annotate ToF checkerboard plane"


def colourise_depth(depth_mm: np.ndarray) -> np.ndarray:
    depth = depth_mm.astype(np.float32)

    valid = (
        np.isfinite(depth)
        & (depth >= 150.0)
        & (depth <= 3000.0)
    )

    output = np.zeros(
        (depth.shape[0], depth.shape[1], 3),
        dtype=np.uint8,
    )

    if np.count_nonzero(valid) < 100:
        return output

    values = depth[valid]

    lower = float(np.percentile(values, 2.0))
    upper = float(np.percentile(values, 98.0))

    if upper <= lower:
        return output

    normalized = np.zeros_like(
        depth,
        dtype=np.uint8,
    )

    scaled = (
        (depth - lower)
        / (upper - lower)
        * 255.0
    )

    normalized[valid] = np.clip(
        scaled[valid],
        0,
        255,
    ).astype(np.uint8)

    output = cv2.applyColorMap(
        255 - normalized,
        cv2.COLORMAP_JET,
    )

    output[~valid] = 0

    return output


class BoardAnnotator:
    def __init__(self) -> None:
        self.sample_folders = sorted(
            folder
            for folder in DATASET_DIR.glob("sample_*")
            if folder.is_dir()
        )

        if not self.sample_folders:
            raise RuntimeError(
                f"No calibration samples found in {DATASET_DIR}"
            )

        self.sample_index = 0
        self.points: list[tuple[int, int]] = []
        self.base_image: np.ndarray | None = None
        self.depth: np.ndarray | None = None

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            960,
            720,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_callback,
        )

    @property
    def current_folder(self) -> Path:
        return self.sample_folders[self.sample_index]

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter,
    ) -> None:
        del flags, parameter

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.points) >= 4:
            return

        self.points.append((x, y))

    def load_current_sample(self) -> None:
        depth_file = self.current_folder / "tof_depth.npy"

        if not depth_file.exists():
            raise FileNotFoundError(
                f"Missing file: {depth_file}"
            )

        self.depth = np.load(depth_file)
        self.base_image = colourise_depth(self.depth)

        annotation_file = (
            self.current_folder
            / ANNOTATION_FILENAME
        )

        if annotation_file.exists():
            polygon = np.load(annotation_file)

            self.points = [
                (int(point[0]), int(point[1]))
                for point in polygon
            ]

            print(
                f"Loaded existing annotation: "
                f"{self.current_folder.name}"
            )
        else:
            self.points = []

    def create_display(self) -> np.ndarray:
        if self.base_image is None:
            raise RuntimeError(
                "No sample image loaded."
            )

        display = self.base_image.copy()

        for index, point in enumerate(
            self.points,
            start=1,
        ):
            cv2.circle(
                display,
                point,
                5,
                (255, 255, 255),
                thickness=-1,
            )

            cv2.putText(
                display,
                str(index),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if len(self.points) >= 2:
            for index in range(
                len(self.points) - 1
            ):
                cv2.line(
                    display,
                    self.points[index],
                    self.points[index + 1],
                    (255, 255, 255),
                    2,
                )

        if len(self.points) == 4:
            polygon = np.asarray(
                self.points,
                dtype=np.int32,
            )

            cv2.line(
                display,
                self.points[-1],
                self.points[0],
                (255, 255, 255),
                2,
            )

            overlay = display.copy()

            cv2.fillPoly(
                overlay,
                [polygon],
                (255, 255, 255),
            )

            display = cv2.addWeighted(
                overlay,
                0.20,
                display,
                0.80,
                0.0,
            )

        title = (
            f"{self.sample_index + 1}/"
            f"{len(self.sample_folders)}  "
            f"{self.current_folder.name}"
        )

        cv2.putText(
            display,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        instructions = (
            "Click 4 board corners | "
            "S save | R reset | N skip | "
            "B back | Q quit"
        )

        cv2.putText(
            display,
            instructions,
            (8, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display

    def save_annotation(self) -> bool:
        if len(self.points) != 4:
            print(
                "Select exactly four board corners "
                "before saving."
            )
            return False

        polygon = np.asarray(
            self.points,
            dtype=np.int32,
        )

        annotation_file = (
            self.current_folder
            / ANNOTATION_FILENAME
        )

        np.save(
            annotation_file,
            polygon,
        )

        overlay = self.create_display()

        cv2.imwrite(
            str(
                self.current_folder
                / OVERLAY_FILENAME
            ),
            overlay,
        )

        print(
            f"Saved: {self.current_folder.name}"
        )

        return True

    def move_next(self) -> bool:
        if self.sample_index >= (
            len(self.sample_folders) - 1
        ):
            print("Reached the final sample.")
            return False

        self.sample_index += 1
        self.load_current_sample()
        return True

    def move_previous(self) -> bool:
        if self.sample_index <= 0:
            print("Already at the first sample.")
            return False

        self.sample_index -= 1
        self.load_current_sample()
        return True

    def run(self) -> None:
        self.load_current_sample()

        print("\nToF board annotation")
        print("Click the four OUTER corners of the board.")
        print("The click order can be clockwise or anticlockwise.")
        print("Press S after four points are selected.\n")

        while True:
            display = self.create_display()

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                self.points = []
                continue

            if key in (ord("s"), ord("S")):
                if self.save_annotation():
                    if not self.move_next():
                        break
                continue

            if key in (ord("n"), ord("N")):
                self.move_next()
                continue

            if key in (ord("b"), ord("B")):
                self.move_previous()
                continue

        cv2.destroyAllWindows()


def main() -> None:
    annotator = BoardAnnotator()
    annotator.run()


if __name__ == "__main__":
    main()