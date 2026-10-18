import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from behavioral_cloning.dataset import read_rows, resolve_image_path
from behavioral_cloning.preprocessing import load_image


@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    steering: float
    session: str = ""
    record: int = 0
    sha256: str = ""
    camera: str = "center"


def expand_cameras(
    examples: list[TrainingExample],
    raw_dir: Path,
    correction: float,
    target_policy: str = "reject",
) -> list[TrainingExample]:
    """Expand already-split training records, auditing every side image.

    Positive steering is right. Out-of-range corrections require an explicit
    clipping policy; default is rejection. Validation remains center-only.
    """
    if not math.isfinite(correction) or not 0 < correction <= 1:
        raise ValueError("Camera correction must be finite and in (0, 1]")
    if target_policy not in {"reject", "clip"}:
        raise ValueError("Unknown camera target policy")
    root = raw_dir.resolve()
    rows_by_session = {}
    expanded = []
    for example in examples:
        session = (root / example.session).resolve()
        if not example.session or not session.is_relative_to(root):
            raise ValueError("Invalid camera source session")
        if example.session not in rows_by_session:
            rows_by_session[example.session] = read_rows(session / "driving_log.csv")
        rows = rows_by_session[example.session]
        if not 1 <= example.record <= len(rows):
            raise ValueError("Invalid camera source record")
        row = rows[example.record - 1]
        if len(row) != 7:
            raise ValueError("Camera source must have seven fields")
        center = resolve_image_path(row[0], session).resolve()
        if center != example.image_path.resolve() or float(row[3]) != example.steering:
            raise ValueError("Camera source does not match manifest")
        if (
            example.sha256
            and hashlib.sha256(center.read_bytes()).hexdigest() != example.sha256
        ):
            raise ValueError("Center image changed since audit")
        expanded.append(example)
        for camera, column, offset in (
            ("left", 1, correction),
            ("right", 2, -correction),
        ):
            target = example.steering + offset
            if not -1 <= target <= 1:
                if target_policy == "reject":
                    raise ValueError(f"Corrected steering outside [-1, 1]: {target}")
                target = max(-1.0, min(1.0, target))
            path = resolve_image_path(row[column].replace("\\", "/"), session).resolve()
            if not path.is_relative_to(session / "IMG"):
                raise ValueError("Side image escapes session IMG directory")
            load_image(path)  # Checks decodability, dimensions, and uint8 pixels.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expanded.append(
                TrainingExample(
                    path, target, example.session, example.record, digest, camera
                )
            )
    return expanded


def load_manifest(
    manifest_path: Path,
    raw_dir: Path,
) -> list[TrainingExample]:
    examples = []
    root = raw_dir.resolve()
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"center", "steering", "session", "record", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {manifest_path}")
        for row in reader:
            image_path = (root / row["center"]).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"Image path escapes raw directory: {image_path}")
            steering = float(row["steering"])
            if not math.isfinite(steering) or not -1 <= steering <= 1:
                raise ValueError(f"Invalid steering: {row['steering']}")
            examples.append(
                TrainingExample(
                    image_path,
                    steering,
                    row["session"],
                    int(row["record"]),
                    row["sha256"],
                )
            )
    if not examples:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return examples


def check_split_overlap(
    train: list[TrainingExample],
    validation: list[TrainingExample],
) -> None:
    """Reject session, file, or exact-image overlap before training."""
    for field in ("session", "image_path", "sha256"):
        left = {getattr(e, field) for e in train if getattr(e, field)}
        right = {getattr(e, field) for e in validation if getattr(e, field)}
        if left & right:
            raise ValueError(f"Training/validation overlap in {field}")
