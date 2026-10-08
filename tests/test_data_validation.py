import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from behavioral_cloning.data_validation import (
    prepare_dataset,
    steering_summary,
    validate_session,
)


def make_session(root: Path, name: str, value: int = 0) -> Path:
    session = root / name
    (session / "IMG").mkdir(parents=True)
    cv2.imwrite(
        str(session / "IMG" / "center.jpg"),
        np.full((160, 320, 3), value, dtype=np.uint8),
    )
    log = session / "driving_log.csv"
    log.write_text("/old/center.jpg,left.jpg,right.jpg,0.2,0,0,10\n")
    return log


def test_reports_all_failures_without_repairing(tmp_path: Path) -> None:
    log = make_session(tmp_path, "session")
    with log.open("a") as stream:
        stream.write("too,few\n")
        stream.write("center.jpg,l,r,nan,0,0,1\n")
        stream.write("center.jpg,l,r,,0,0,1\n")
        stream.write("missing.jpg,l,r,0,0,0,1\n")
        stream.write("broken.jpg,l,r,0,0,0,1\n")
        stream.write("center.jpg,l,r,1.1,0,0,1\n")
        stream.write("center.jpg,l,r,0,bad,0,1\n")
    (log.parent / "IMG" / "broken.jpg").write_text("not an image")
    before = log.read_bytes()
    result = validate_session(log)
    assert result["total_records"] == 8
    assert result["usable_records"] == 2
    assert {i["record"] for i in result["issues"]} == set(range(2, 9))
    assert result["issues"][-1]["severity"] == "warning"
    assert log.read_bytes() == before


def test_split_reproducibility_and_duplicate_detection(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_session(raw, "a", 0)
    make_session(raw, "b", 255)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train": ["a"], "validation": ["b"]}))
    output = tmp_path / "output"
    assert prepare_dataset(raw, config, output)
    first = (output / "train.csv").read_bytes()
    assert prepare_dataset(raw, config, output)
    assert first == (output / "train.csv").read_bytes()
    with (output / "validation.csv").open() as stream:
        assert next(csv.DictReader(stream))["session"] == "b"
    (raw / "b/IMG/center.jpg").write_bytes((raw / "a/IMG/center.jpg").read_bytes())
    assert not prepare_dataset(raw, config, output)
    assert not (output / "train.csv").exists()
    assert json.loads((output / "report.json").read_text())[
        "cross_split_duplicate_images"
    ]
    config.write_text(json.dumps({"train": ["a"], "validation": ["a"]}))
    with pytest.raises(ValueError, match="overlap"):
        prepare_dataset(raw, config, output)


def test_exclusions_and_empty_session(tmp_path: Path) -> None:
    log = make_session(tmp_path, "session")
    with log.open("a") as stream:
        stream.write("bad,row\n")
    result = validate_session(log, {2})
    assert result["usable_records"] == 1
    assert result["excluded_records"] == 1
    assert not result["issues"]
    assert validate_session(log, {3})["issues"][-1]["field"] == "exclusion"
    log.write_text("")
    assert validate_session(log)["issues"]


def test_distribution_boundaries() -> None:
    result = steering_summary([-1, -0.5, -0.05, 0, 0.05, 0.5, 1])
    assert list(result["bins"].values()) == [1, 1, 3, 1, 1]
    assert result["near_straight_fraction"] == 3 / 7
    assert steering_summary([])["mean"] is None
