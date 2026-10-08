"""Audit center-image/steering examples without modifying raw recordings."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from behavioral_cloning.dataset import resolve_image_path


def steering_summary(values: list[float]) -> dict[str, Any]:
    """Use explicit bins; near straight means absolute steering <= 0.05."""
    counts = {
        "[-1,-0.5)": sum(v < -0.5 for v in values),
        "[-0.5,-0.05)": sum(-0.5 <= v < -0.05 for v in values),
        "[-0.05,0.05]": sum(abs(v) <= 0.05 for v in values),
        "(0.05,0.5]": sum(0.05 < v <= 0.5 for v in values),
        "(0.5,1]": sum(v > 0.5 for v in values),
    }
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "zero_predictor_mse": sum(v * v for v in values) / len(values)
        if values
        else None,
        "exact_zero_count": values.count(0.0),
        "near_straight_fraction": counts["[-0.05,0.05]"] / len(values)
        if values
        else None,
        "bins": counts,
    }


def validate_session(
    csv_path: Path, excluded: set[int] | None = None
) -> dict[str, Any]:
    """Record numbers are one-based CSV records (the files have no header)."""
    excluded = excluded or set()
    issues: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    shapes: Counter[str] = Counter()
    total = 0

    def issue(record: int, field: str, message: str, severity: str = "error") -> None:
        issues.append(
            dict(record=record, field=field, message=message, severity=severity)
        )

    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for total, row in enumerate(csv.reader(stream, strict=True), start=1):
                if total in excluded:
                    continue
                if len(row) != 7:
                    issue(total, "row", f"Expected 7 fields; found {len(row)}")
                    continue
                steering: float | None = None
                for index, field in enumerate(
                    ("steering", "throttle", "brake", "speed"), start=3
                ):
                    try:
                        value = float(row[index])
                        if not math.isfinite(value):
                            raise ValueError("Value must be finite")
                        if field == "steering":
                            if not -1 <= value <= 1:
                                raise ValueError("Steering must be in [-1, 1]")
                            steering = value
                    except ValueError as exc:
                        issue(
                            total,
                            field,
                            str(exc),
                            "error" if field == "steering" else "warning",
                        )
                image_path = resolve_image_path(row[0], csv_path.parent)
                if not row[0].strip() or not image_path.is_file():
                    issue(total, "center", f"Missing image: {image_path}")
                    continue
                try:
                    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                except cv2.error:
                    image = None
                if image is None:
                    issue(total, "center", f"Cannot decode: {image_path}")
                    continue
                shapes[str(tuple(image.shape))] += 1
                if image.shape != (160, 320, 3) or str(image.dtype) != "uint8":
                    issue(
                        total, "center", "Expected uint8 image of shape (160, 320, 3)"
                    )
                    continue
                if steering is not None:
                    examples.append(
                        {
                            "session": csv_path.parent.name,
                            "record": total,
                            "center": image_path.relative_to(
                                csv_path.parent.parent
                            ).as_posix(),
                            "steering": steering,
                            "sha256": hashlib.sha256(
                                image_path.read_bytes()
                            ).hexdigest(),
                        }
                    )
    except (OSError, UnicodeError, csv.Error) as exc:
        issue(0, "csv", str(exc))
    if total == 0:
        issue(0, "csv", "Session has no records")
    for record in sorted(excluded):
        if not 1 <= record <= total:
            issue(record, "exclusion", "Excluded record does not exist")
    if not examples:
        issue(0, "session", "Session has no usable examples")
    return {
        "total_records": total,
        "excluded_records": len(excluded.intersection(range(1, total + 1))),
        "usable_records": len(examples),
        "shapes": dict(shapes),
        "issues": issues,
        "steering": steering_summary([e["steering"] for e in examples]),
        "examples": examples,
    }


def prepare_dataset(raw_dir: Path, config_path: Path, output_dir: Path) -> bool:
    """Always write diagnostics; publish manifests only if the audit passes."""
    if output_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("Output directory must be outside the raw recordings")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assignments = {split: config[split] for split in ("train", "validation")}
    if any(not isinstance(group, list) or not group for group in assignments.values()):
        raise ValueError("Both splits must be nonempty lists")
    names = [name for group in assignments.values() for name in group]
    if any(
        not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n
        for n in names
    ):
        raise ValueError("Session names must be plain directory names")
    if len(names) != len(set(names)):
        raise ValueError("Each session must appear exactly once, with no split overlap")
    exclusions: dict[str, set[int]] = {name: set() for name in names}
    for entry in config.get("exclusions", []):
        name = entry["session"]
        start, end = entry["start_record"], entry["end_record"]
        if name not in exclusions or not entry["reason"].strip():
            raise ValueError("Exclusions require an assigned session and a reason")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
            raise ValueError("Exclusions require a positive inclusive record range")
        exclusions[name].update(range(start, end + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale generated manifests before auditing changed data.
    for split in assignments:
        (output_dir / f"{split}.csv").unlink(missing_ok=True)
    sessions = {}
    split_examples: dict[str, list[dict[str, Any]]] = {}
    for split, group in assignments.items():
        split_examples[split] = []
        for name in group:
            result = validate_session(
                raw_dir / name / "driving_log.csv", exclusions[name]
            )
            split_examples[split].extend(result.pop("examples"))
            sessions[name] = result
            print(
                f"{name}: {result['usable_records']}/{result['total_records']} usable"
            )
    train_hashes = {e["sha256"] for e in split_examples["train"]}
    duplicates = [
        e for e in split_examples["validation"] if e["sha256"] in train_hashes
    ]
    passed = not duplicates and not any(
        issue["severity"] == "error"
        for result in sessions.values()
        for issue in result["issues"]
    )
    report = {
        "passed": passed,
        "config": config,
        "unassigned_sessions": sorted(
            p.name for p in raw_dir.iterdir() if p.is_dir() and p.name not in names
        ),
        "sessions": sessions,
        "cross_split_duplicate_images": duplicates,
        "splits": {
            s: steering_summary([e["steering"] for e in examples])
            for s, examples in split_examples.items()
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if passed:
        for split, examples in split_examples.items():
            with (output_dir / f"{split}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["session", "record", "center", "steering", "sha256"],
                )
                writer.writeheader()
                writer.writerows(examples)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset"))
    args = parser.parse_args()
    passed = prepare_dataset(args.raw_dir, args.config, args.output_dir)
    print(
        f"Audit {'passed' if passed else 'FAILED'}; "
        f"see {args.output_dir / 'report.json'}"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
