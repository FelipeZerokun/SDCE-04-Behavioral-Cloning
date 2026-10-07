from pathlib import Path

from behavioral_cloning.dataset import parse_row, read_rows, resolve_image_path


def test_resolve_image_path():
    recorded_path = "home/user/training_data/session/IMG/center.jpg"
    session_dir = Path("data/raw/session")

    result = resolve_image_path(recorded_path, session_dir)

    assert result == session_dir / "IMG" / "center.jpg"

def test_read_rows(tmp_path: Path):
    csv_path = tmp_path / "driving_log.csv"
    csv_path.write_text(
        "center1.jpg,left1.jpg,right1.jpg,0,0.5,0,10\n"
        "center2.jpg,left2.jpg,right2.jpg,-0.2,0.3,0,12\n",
        encoding="utf-8",
    )

    rows = read_rows(csv_path)

    assert rows == [
        ["center1.jpg" ,"left1.jpg","right1.jpg","0","0.5","0","10"],
        ["center2.jpg","left2.jpg","right2.jpg","-0.2","0.3","0","12"],
    ]

def test_parse_row():
    row = [
        "/home/user/IMG/center.jpg",
        "/home/user/IMG/left.jpg",
        "/home/user/IMG/right.jpg",
        "-0.25",
        "0.6",
        "0.1",
        "1.031372E-05",
    ]
    session_dir = Path("data/raw/session")

    sample = parse_row(row, session_dir)

    assert sample.center == session_dir / "IMG" / "center.jpg"
    assert sample.left == session_dir / "IMG" / "left.jpg"
    assert sample.right == session_dir / "IMG" / "right.jpg"
    assert sample.steering == -0.25
    assert sample.throttle == 0.6
    assert sample.brake == 0.1
    assert sample.speed == 1.031372e-05