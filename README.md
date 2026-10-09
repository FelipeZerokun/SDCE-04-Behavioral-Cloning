# Behavioral Cloning — Python

A behavioral cloning project in development, with the goal of learning
steering commands from recorded camera images and driving in the Udacity simulator.
The project uses UV for reproducible Python environments and dependency
management.

This repository modernizes the Behavioral Cloning project from Udacity's
Self-Driving Car Engineer Nanodegree as a typed, tested Python package
that can be run as Python modules.

**Status:** Dataset loading, the interactive viewer, and dataset validation with
reproducible split manifests are implemented. All 8,704 current center-image
examples passed the audit. Training and offline evaluation are implemented and
smoke-tested. Full baseline training and simulator inference remain outstanding.
See [dataset preparation](docs/dataset.md) for results and the preprocessing contract.

**Next milestone:** Run full baseline training and assess held-out steering
predictions. See [training instructions](docs/training.md).

The active development branch is `modern/keras`. Keras and TensorFlow are locked
project dependencies. The pipeline saves native `.keras` and full-model `.h5`
files and verifies their predictions after reload in the current environment.

## Features

- Load recording sessions into `DrivingSample` objects with three image paths
  and numeric steering, throttle, brake, and speed values.
- Resolve recorded Ubuntu paths to each local session's `IMG` folder without
  modifying the original CSV.
- Inspect synchronized left, center, and right camera images in an OpenCV window.
- Display frame number and telemetry, with keyboard navigation and window-close handling.
- Test CSV reading, image-path resolution, and row parsing using synthetic data.
- Audit center-image examples and generate reproducible session split manifests
  with steering summaries and exact image-file duplicate checks.
- Prepare RGB float32 images and load batches from audited manifests.
- Train a Keras CNN with early stopping and best-checkpoint selection.
- Export and reload-check `.keras` and `.h5` models, and generate validation
  metrics, prediction records, loss curves, and example error plots.

## Training and driving pipeline

The training input is a center-camera image; the target is its
recorded steering command. Throttle is displayed for inspection but is not
a training target. Each example uses one image, not a sequence of images.
Side-camera inputs and corrected steering labels are deferred.

1. Collect recordings and inspect them with the viewer (available).
2. Assign entire sessions to training and validation sets before augmentation
   to reduce leakage between similar neighboring frames (implemented).
3. Apply the documented preprocessing and train a baseline network (implemented).
4. Compare training and validation mean squared error (implemented).
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
testing before augmentation. The first baseline uses no augmentation. A later
training-only horizontal-flipping experiment must invert the steering sign.

Raw recordings are excluded from Git.

### Current recording inventory and collection history

All current recordings use the simulator's normal track, not the advanced track.
The simulator's initial driving direction is counterclockwise (`normal`);
`inverse` means clockwise. Lap direction does not imply steering direction:
the circuit includes both left and right turns.

The following inventory and split were agreed on 2026-10-08. The recorder
confirmed visual quality with no exclusions. All included center images and
steering targets passed the automated audit; see [the results](docs/dataset.md).

| Session | Purpose | Split |
| --- | --- | --- |
| `normal_lap_01` | One complete, independently driven counterclockwise lap | Training |
| `normal_lap_02` | One complete, independently driven counterclockwise lap | Validation |
| `inverse_lap_01` | One complete, independently driven clockwise lap | Training |
| `inverse_lap_02` | One complete, independently driven clockwise lap | Validation |
| `recover_left` | Recover toward the center from too far left, with rightward correction | Training |
| `recover_right` | Recover toward the center from too far right, with leftward correction | Training |
| `sharp_curve_left` | Selected sharp left turns from the driver's perspective | Training |
| `sharp_curve_right` | Selected sharp right turns from the driver's perspective | Training |
| `slight_curve_left` | Selected gentle left turns from the driver's perspective | Training |
| `slight_curve_right` | Selected gentle right turns from the driver's perspective | Training |
| `bridge` | Selected bridge crossings in both directions | Training |

Bridge and curve recordings contain selected passes rather than complete laps.
Collection included repeated passes and driving in both directions, stopping
recording to turn around or reposition. Consequently, a folder and its CSV may
contain multiple recording intervals, not one continuous drive. Curve names
describe the turn as seen by the driver, not a fixed location on the track.

Recovery recordings start with the car already outside the desired position,
capture the return toward the center, and continue briefly afterward. Deliberate
positioning is not part of the recorded recovery. The recorder reports discarding
recordings with crashes, major mistakes, or complete accidental departures from
the track, intending to retain correct driving and recovery demonstrations.
Visual quality was confirmed by the recorder; the automated audit checks sample
integrity, not driving correctness.
The simulator version has not yet been documented.

For the single-image baseline, recording interruptions do not invalidate individual
image–steering pairs. Keep the existing folders and original recordings intact.
Any future sequence-based model would need verified continuous-segment boundaries
so that input windows cannot cross interruptions; those boundaries have not been
identified. Separate directories per continuous take are recommended for future
collection.

### Dataset preparation decisions and results

Keep each complete session folder in one split, before augmentation. The agreed
validation set contains an independent full lap in each direction; training
contains the other two laps plus all targeted demonstrations. The `01`/`02`
choice is a convention, not a measured quality ranking.
An exact 80/20 sample ratio is not required. No separate test set is assigned yet.

Validation will estimate steering prediction on held-out recordings of the same
track. It does not establish performance on the advanced track, directly evaluate
recovery ability, or replace autonomous simulator evaluation.

Visual review is complete according to the recorder, with no exclusions identified.
For future reviews, record suspicious frame ranges and reasons separately; do not
remove raw images or edit the original CSVs. Off-center positioning alone is not
a reason to reject an intentional recovery example.

The milestone is complete when:

- The session inventory includes reviewed quality findings and any exclusions.
- Every included center image decodes successfully and every steering target is
  valid and finite; sample integrity checks are recorded.
- Split membership and exclusions are reproducible, with no session overlap.
- Training and validation steering distributions are summarized, including the
  proportion of straight or near-straight driving and its defined threshold.
- The future model's input format, crop, resize, color order, data type, and
  normalization are documented for identical use in training and inference.

These milestone criteria are now met for the current recordings. Run the audit
again whenever recordings or assignments change:

```powershell
uv run python -m behavioral_cloning.data_validation
```

`configs/dataset.json` records membership and exclusions. The command writes
`outputs/dataset/report.json` and, on success, `train.csv` and `validation.csv`.
See [dataset preparation](docs/dataset.md) for checks, results, and preprocessing.

### Inspect a recording

Run commands from the repository root so relative data paths resolve correctly:

```powershell
uv run python -m behavioral_cloning.dataset
uv run python -m behavioral_cloning.viewer
```

The dataset module prints the sample count and first parsed sample. The viewer
shows the three cameras side by side with a telemetry header.

The dataset module selects `normal_lap_01`; the viewer currently selects
`recover_left` under `data/raw/` in their `__main__` blocks.
To inspect another session, change `csv_path` in the module
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

```powershell
uv run python -m behavioral_cloning.train --output-dir outputs/baseline-01
```

See [training instructions](docs/training.md) for smoke mode, settings and outputs.

### Evaluate a model

```powershell
uv run python -m behavioral_cloning.evaluate outputs/baseline-01/model.keras --manifest outputs/baseline-01/dataset/validation.csv
```

### Drive in the simulator

Not implemented yet. The scripts under `references/udacity/` are historical
references, not the current package's inference implementation.

## Configuration

`pyproject.toml` declares dependencies and configures pytest, Ruff, and mypy.
`uv.lock` records resolved dependencies. Viewer and loader session paths are
configured directly in their executable blocks. The validation command accepts
CLI paths and uses `configs/dataset.json` for split assignments and exclusions.
Training accepts CLI options for epochs, batch size, learning rate, patience,
seed, and output paths. Each run saves its configuration and fresh audited
manifests. See [training configuration and artifacts](docs/training.md).

## Development

Run the complete test suite:

```powershell
uv run python -m pytest -q
```

All 18 tests passed on 2026-10-09. They cover CSV loading and dataset auditing,
image preprocessing and decoding, manifest validation, batch alignment, model
learning, model exports, and evaluation metrics using synthetic data. Ruff and
strict mypy passed for the six new pipeline modules. A separate real-recording
smoke run audited all 8,704 examples, trained on 32 examples, and evaluated 16;
both exports passed prediction agreement checks after reload. This verifies the
pipeline, not full-dataset model quality. The viewer's A/D boundaries, Q/Escape
exit, and X-button exit were previously checked manually on Windows.

## Project structure

```text
src/behavioral_cloning/
    __init__.py
    dataset.py          # DrivingSample, CSV reading, path resolution, session loading
    data_validation.py  # Integrity audit, split manifests, steering summaries
    viewer.py           # Sample rendering and OpenCV navigation loop
    preprocessing.py    # Image decoding, RGB conversion and normalization
    training_data.py    # Manifest loading and split overlap checks
    batches.py          # Keras image batches and training-only shuffling
    model.py            # Small steering CNN and optimizer
    train.py            # Audit, training, checkpoint selection and exports
    evaluate.py         # Validation metrics, predictions and plots
tests/
    test_dataset.py     # Unit tests using synthetic inputs
    test_data_validation.py
    test_preprocessing.py
    test_training.py
configs/dataset.json    # Session assignments and separate exclusion ranges
docs/dataset.md         # Audit results and preprocessing contract
docs/training.md        # Training commands, design and verified status
references/udacity/     # Historical drive.py, video.py, and their license
data/raw/               # Local recordings (Git-ignored)
outputs/                # Local audits, models and reports (Git-ignored)
pyproject.toml          # Project dependencies and development configuration
uv.lock                 # Dependency lockfile
```

## Limitations

- The loader expects a nonempty, headerless CSV with seven columns in this order:
  `center,left,right,steering,throttle,brake,speed`. Use the separate validation
  command for diagnostics before training; the viewer loader remains simple.
- Path resolution interprets Unix-style recorded paths and uses their filenames
  under the local `IMG` folder; Windows-style recorded paths are not yet supported.
- Missing or unreadable images stop the viewer with an error. Camera images must
  have matching heights and compatible types for horizontal concatenation.
- Full baseline training has not yet run. Augmentation and simulator integration
  remain planned; training, evaluation and model saving are implemented.
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
