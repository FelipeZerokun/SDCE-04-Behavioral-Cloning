"""Preserve source and input identities for each new training configuration."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from behavioral_cloning.training_data import TrainingExample


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_provenance(
    output: Path,
    dataset_config: Path,
    train: list[TrainingExample],
    validation: list[TrainingExample],
) -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = output / "source"
    source_dir.mkdir()
    identities = {}
    for path in [
        *Path(__file__).parent.glob("*.py"),
        root / "pyproject.toml",
        root / "uv.lock",
    ]:
        relative = path.relative_to(root)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        identities[str(relative)] = file_hash(path)
    for name, args in (
        ("git-head.txt", ["rev-parse", "HEAD"]),
        ("git-status.txt", ["status", "--porcelain"]),
        ("git-diff.patch", ["diff", "HEAD", "--binary"]),
    ):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False
        )
        (source_dir / name).write_bytes(result.stdout + result.stderr)
    (output / "dataset-config.json").write_bytes(dataset_config.read_bytes())
    identities["dataset-config.json"] = file_hash(dataset_config)
    for split, examples in (("train", train), ("validation", validation)):
        records = [
            {**asdict(example), "image_path": str(example.image_path)}
            for example in examples
        ]
        path = output / "dataset" / f"{split}-effective.json"
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        identities[str(path.relative_to(output))] = file_hash(path)
        manifest = output / "dataset" / f"{split}.csv"
        identities[str(manifest.relative_to(output))] = file_hash(manifest)
    (output / "provenance.json").write_text(
        json.dumps(identities, indent=2) + "\n", encoding="utf-8"
    )
