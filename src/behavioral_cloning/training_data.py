import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    steering: float
    session: str = ""
    record: int = 0
    sha256: str = ""


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
