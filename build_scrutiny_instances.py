#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
from bisect import bisect_left
from pathlib import Path


IMAGE_FOLDERS = {
    "zed_rgb": "zed_rgb",
    "zed_depth": "zed_depth",
    "tof_depth_raw": "tof_depth_raw",
    "tof_depth8": "tof_depth8",
    "fused_overlay": "fused_overlay",
}

MEASUREMENT_CSV = "diameter_measurements.csv"


def parse_timestamp_from_name(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"Cannot parse timestamp from: {path.name}")

    return int(parts[-1])


def load_image_index(folder: Path):
    files = sorted(folder.glob("*.png"))

    entries = []
    for file_path in files:
        try:
            timestamp_ns = parse_timestamp_from_name(file_path)
        except Exception:
            continue

        entries.append((timestamp_ns, file_path))

    timestamps = [item[0] for item in entries]
    return timestamps, entries


def nearest_entry(target_ns: int, timestamps, entries):
    if not entries:
        return None

    position = bisect_left(timestamps, target_ns)

    candidates = []

    if position < len(entries):
        candidates.append(entries[position])

    if position > 0:
        candidates.append(entries[position - 1])

    return min(
        candidates,
        key=lambda item: abs(item[0] - target_ns),
    )


def load_measurements(csv_path: Path):
    rows = []

    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "relative_time_s": float(row["relative_time_s"]),
                    "diameter_mm": float(row["diameter_mm"]),
                }
            )

    return rows


def process_session(
    session_dir: Path,
    output_root: Path,
    max_delta_ms: float,
    copy_mode: str,
):
    session_name = session_dir.name
    if session_name.endswith("_export"):
        base_session_name = session_name[:-7]
    else:
        base_session_name = session_name

    measurement_csv = session_dir / MEASUREMENT_CSV

    if not measurement_csv.exists():
        return {
            "session": base_session_name,
            "status": "skipped_no_measurement_csv",
            "measurement_rows": 0,
            "complete_instances": 0,
        }

    measurements = load_measurements(measurement_csv)

    image_indexes = {}
    for key, folder_name in IMAGE_FOLDERS.items():
        folder = session_dir / folder_name

        if folder.exists():
            image_indexes[key] = load_image_index(folder)
        else:
            image_indexes[key] = ([], [])

    session_output = output_root / base_session_name
    session_output.mkdir(parents=True, exist_ok=True)

    max_delta_ns = int(max_delta_ms * 1_000_000)

    summary_rows = []
    complete_count = 0
    missing_counts = {key: 0 for key in IMAGE_FOLDERS}

    for index, measurement in enumerate(measurements, start=1):
        measurement_ts = measurement["timestamp_ns"]

        matches = {}
        deltas_ms = {}
        complete = True

        for key in IMAGE_FOLDERS:
            timestamps, entries = image_indexes[key]
            nearest = nearest_entry(
                measurement_ts,
                timestamps,
                entries,
            )

            if nearest is None:
                matches[key] = None
                deltas_ms[key] = None
                missing_counts[key] += 1
                complete = False
                continue

            image_ts, image_path = nearest
            delta_ns = abs(image_ts - measurement_ts)
            delta_ms = delta_ns / 1_000_000.0

            if delta_ns > max_delta_ns:
                matches[key] = None
                deltas_ms[key] = delta_ms
                missing_counts[key] += 1
                complete = False
            else:
                matches[key] = image_path
                deltas_ms[key] = delta_ms

        instance_name = f"instance_{index:06d}"

        if complete:
            instance_dir = session_output / instance_name
            instance_dir.mkdir(parents=True, exist_ok=True)

            for key, source_path in matches.items():
                destination = instance_dir / f"{key}.png"

                if copy_mode == "copy":
                    shutil.copy2(source_path, destination)
                elif copy_mode == "hardlink":
                    try:
                        destination.hardlink_to(source_path)
                    except Exception:
                        shutil.copy2(source_path, destination)
                elif copy_mode == "symlink":
                    destination.symlink_to(source_path.resolve())

            metadata = {
                "instance_id": instance_name,
                "session": base_session_name,
                "measurement_timestamp_ns": measurement_ts,
                "relative_time_s": measurement["relative_time_s"],
                "diameter_mm": measurement["diameter_mm"],
                "matched_files": {
                    key: str(path)
                    for key, path in matches.items()
                },
                "time_deltas_ms": deltas_ms,
            }

            with (instance_dir / "measurement.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(metadata, file, indent=2)

            complete_count += 1

        summary_row = {
            "session": base_session_name,
            "instance_index": index,
            "measurement_timestamp_ns": measurement_ts,
            "relative_time_s": measurement["relative_time_s"],
            "diameter_mm": measurement["diameter_mm"],
            "complete": complete,
        }

        for key in IMAGE_FOLDERS:
            summary_row[f"{key}_file"] = (
                str(matches[key]) if matches[key] else ""
            )
            summary_row[f"{key}_delta_ms"] = (
                deltas_ms[key]
                if deltas_ms[key] is not None
                else ""
            )

        summary_rows.append(summary_row)

    summary_csv = session_output / "session_instances.csv"

    fieldnames = [
        "session",
        "instance_index",
        "measurement_timestamp_ns",
        "relative_time_s",
        "diameter_mm",
        "complete",
    ]

    for key in IMAGE_FOLDERS:
        fieldnames.extend(
            [
                f"{key}_file",
                f"{key}_delta_ms",
            ]
        )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    session_summary = {
        "session": base_session_name,
        "status": "processed",
        "measurement_rows": len(measurements),
        "complete_instances": complete_count,
        "incomplete_instances": len(measurements) - complete_count,
        "missing_counts": missing_counts,
        "max_delta_ms": max_delta_ms,
    }

    with (session_output / "session_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(session_summary, file, indent=2)

    return session_summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create timestamp-matched Scrutiny measurement instances "
            "from exported bag folders."
        )
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "Folder containing scrutiny_*_export session folders, "
            "for example ~/Scrutiny_Dataset"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output root. Default: <dataset_root>/Matched_Instances"
        ),
    )

    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=100.0,
        help=(
            "Maximum allowed timestamp difference for each matched image. "
            "Default: 100 ms"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help=(
            "How to place matched images into instance folders. "
            "Default: hardlink"
        ),
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else dataset_root / "Matched_Instances"
    )

    output_root.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
        and path.name.startswith("scrutiny_")
        and path.name.endswith("_export")
    )

    all_summaries = []

    for session_dir in session_dirs:
        print(f"Processing {session_dir.name}...")

        summary = process_session(
            session_dir=session_dir,
            output_root=output_root,
            max_delta_ms=args.max_delta_ms,
            copy_mode=args.mode,
        )

        all_summaries.append(summary)

        print(
            f"  measurements: {summary.get('measurement_rows', 0)} | "
            f"complete: {summary.get('complete_instances', 0)}"
        )

    total_measurements = sum(
        item.get("measurement_rows", 0)
        for item in all_summaries
    )

    total_complete = sum(
        item.get("complete_instances", 0)
        for item in all_summaries
    )

    skipped_sessions = [
        item["session"]
        for item in all_summaries
        if item.get("status") != "processed"
    ]

    overall_summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "session_count": len(session_dirs),
        "processed_session_count": sum(
            1
            for item in all_summaries
            if item.get("status") == "processed"
        ),
        "skipped_sessions": skipped_sessions,
        "total_measurement_rows": total_measurements,
        "total_complete_instances": total_complete,
        "total_incomplete_instances": (
            total_measurements - total_complete
        ),
        "max_delta_ms": args.max_delta_ms,
        "mode": args.mode,
        "sessions": all_summaries,
    }

    with (output_root / "overall_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(overall_summary, file, indent=2)

    with (output_root / "overall_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "session",
            "status",
            "measurement_rows",
            "complete_instances",
            "incomplete_instances",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(all_summaries)

    print()
    print("Finished")
    print(f"Sessions found: {len(session_dirs)}")
    print(f"Measurement rows: {total_measurements}")
    print(f"Complete instances: {total_complete}")
    print(
        f"Incomplete instances: "
        f"{total_measurements - total_complete}"
    )
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
