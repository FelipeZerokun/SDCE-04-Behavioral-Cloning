import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass
class DrivingSample:
    center: Path
    left: Path
    right: Path
    steering: float
    throttle: float
    brake: float
    speed: float


def read_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        return list(reader)


def resolve_image_path(recorded_path: str, session_dir: Path) -> Path:
    filename = PurePosixPath(recorded_path.strip()).name
    return session_dir / "IMG" / filename


def parse_row(row: list[str], session_dir: Path) -> DrivingSample:
    return DrivingSample(
        center=resolve_image_path(row[0], session_dir),
        left=resolve_image_path(row[1], session_dir),
        right=resolve_image_path(row[2], session_dir),
        steering=float(row[3]),
        throttle=float(row[4]),
        brake=float(row[5]),
        speed=float(row[6]),
    )


def load_session(csv_path: Path) -> list[DrivingSample]:
    rows = read_rows(csv_path)
    session_dir = csv_path.parent

    return [parse_row(row, session_dir) for row in rows]


if __name__ == "__main__":
    csv_path = Path("data/raw/normal_lap_01/driving_log.csv")
    samples = load_session(csv_path)

    print(f"Number of samples: {len(samples)}")
    print(samples[0])
