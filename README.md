# Behavioral Cloning — Python

A behavioral cloning project in development, with the goal of learning
steering commands from recorded camera images and driving in the Udacity simulator.
The project uses UV for reproducible Python environments and dependency
management.

This repository modernizes the Behavioral Cloning project from Udacity's
Self-Driving Car Engineer Nanodegree as a typed, tested Python package
that can be run as Python modules.

**Status:** Dataset loading and the interactive three-camera viewer are
implemented. Training and simulator inference are not yet implemented.

**Next milestone:** Prepare training and validation data, then build and
train a baseline steering-prediction network using modern Keras, following Udacity's
model-loading workflow and targeting a full-model `model.h5` export. The goal
is to modernize project structure, testing, and data preparation while keeping
changes to the original driving script minimal.

The active development branch is `modern/keras`. Keras and its execution
backend will be configured when training is introduced; they are not yet
installed as project dependencies. We plan to retain a native `.keras` model
for development and verify a full-model `.h5` export for Udacity compatibility.

## Features

- Load recording sessions into `DrivingSample` objects with three image paths
  and numeric steering, throttle, brake, and speed values.
- Resolve recorded Ubuntu paths to each local session's `IMG` folder without
  modifying the original CSV.
- Inspect synchronized left, center, and right camera images in an OpenCV window.
- Display frame number and telemetry, with keyboard navigation and window-close handling.
- Test CSV reading, image-path resolution, and row parsing using synthetic data.

## Training and driving pipeline

The planned training input is a center-camera image; the target is its
recorded steering angle. Throttle is displayed for inspection but is not
a training target.

1. Collect recordings and inspect them with the viewer (available).
2. Assign entire sessions to training and validation sets before augmentation
   to reduce leakage between similar neighboring frames (next).
3. Define preprocessing and build a baseline network (planned).
4. Compare training and validation mean squared error (planned).
5. Connect the model to the simulator and evaluate autonomous driving (planned).

A low validation error alone does not establish successful driving; the
model must also keep the vehicle on the road in the simulator.

## Requirements

- Python 3.12 and UV.
- A desktop environment for OpenCV windows.
- Local simulator recordings for viewing, with `driving_log.csv` and `IMG/`
  together in each session directory.

## Installation

From the repository root, install the locked project and development dependencies:

```powershell
uv sync --locked
```

OpenCV is included in the project dependencies. Recordings and the simulator
are local resources and are not included in Git.

## Usage

### Collect training data

Launch the simulator and enter Training Mode. Practice driving before
starting a recording.

1. Choose a separate output directory for each recording session.
2. Position the car near the center of the road.
3. Start recording and drive smoothly for a few laps.
4. Stop recording before resetting or repositioning the vehicle.
5. Verify that the session contains `driving_log.csv` and an `IMG` folder.

Begin with a short test recording to verify the output before collecting
full sessions.

Collect center-road driving in both directions and separate recovery
demonstrations from the left and right sides. During recovery collection,
record the return to the center, rather than the deliberate departure
from it.

Keep recordings organized by session, for example:

    data/raw/track1_center_clockwise_001/
    data/raw/track1_center_counterclockwise_001/
    data/raw/track1_recovery_left_001/
    data/raw/track1_recovery_right_001/

Document each session's simulator version, track, direction, driving
purpose, and any mistakes.

Entire recording sessions will be assigned to training, validation, or
testing before augmentation. Horizontal flipping will be performed in
the training pipeline, with the steering sign inverted.

Raw recordings are excluded from Git.

### Inspect a recording

Run commands from the repository root so relative data paths resolve correctly:

```powershell
uv run python -m behavioral_cloning.dataset
uv run python -m behavioral_cloning.viewer
```

The dataset module prints the sample count and first parsed sample. The viewer
shows the three cameras side by side with a telemetry header.

Both modules currently select `data/raw/normal_lap_01/driving_log.csv` in their
`__main__` blocks. To inspect another session, change `csv_path` in the module
you are running. There is no command-line session argument yet.

With the viewer window focused, use:

| Control | Action |
| --- | --- |
| `a` | Previous sample; stops at the first frame |
| `d` | Next sample; stops at the last frame |
| `q` or Escape | Exit |
| Window X button | Exit |

Letter controls currently use lowercase. Images are redrawn only when the
selected sample changes.

### Train a model

Not implemented yet; this is the next development milestone after data preparation.

### Evaluate a model

Not implemented yet.

### Drive in the simulator

Not implemented yet. The scripts under `references/udacity/` are historical
references, not the current package's inference implementation.

## Configuration

`pyproject.toml` declares dependencies and configures pytest, Ruff, and mypy.
`uv.lock` records resolved dependencies. The session path is currently configured
directly in each executable module as described above.

## Development

Run the dataset tests:

```powershell
uv run python -m pytest tests/test_dataset.py -v
```

The three tests cover path resolution, preservation of CSV row order and string
fields, and conversion to a `DrivingSample`. They do not require real recordings.
All three were confirmed passing during development. The viewer's A/D boundaries,
Q/Escape exit, and X-button exit were manually checked on Windows.

## Project structure

```text
src/behavioral_cloning/
    __init__.py
    dataset.py          # DrivingSample, CSV reading, path resolution, session loading
    viewer.py           # Sample rendering and OpenCV navigation loop
tests/
    test_dataset.py     # Unit tests using synthetic inputs
references/udacity/     # Historical drive.py, video.py, and their license
data/raw/               # Local recordings (Git-ignored)
pyproject.toml          # Project dependencies and development configuration
uv.lock                 # Dependency lockfile
```

## Limitations

- The loader expects a nonempty, headerless CSV with seven columns in this order:
  `center,left,right,steering,throttle,brake,speed`. Comprehensive dataset validation
  and friendly diagnostics for malformed rows are not implemented.
- Path resolution interprets Unix-style recorded paths and uses their filenames
  under the local `IMG` folder; Windows-style recorded paths are not yet supported.
- Missing or unreadable images stop the viewer with an error. Camera images must
  have matching heights and compatible types for horizontal concatenation.
- Training, augmentation, model saving, and simulator integration remain planned.
- Keras is the selected training API. Compatibility between the exported model,
  modern dependencies, and the legacy driving script still needs verification;
  using `.h5` alone does not guarantee the old script will run unchanged.

## Legacy project and attribution

The original `drive.py` and `video.py` scripts are preserved in
[`references/udacity/`](references/udacity/README.md), along with their
Udacity MIT license. They provide references for simulator communication
and video generation as the modern pipeline is built.

## License

See [LICENSE](LICENSE). The preserved Udacity scripts include their own
[MIT license](references/udacity/LICENSE).
