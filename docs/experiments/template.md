# Experiment: <run-id>

Status: proposed / implemented / smoke-tested / trained / driving-tested / concluded.
Use "not measured" or "not recorded" for missing evidence; never infer success.

## Question and comparison

- Hypothesis and observed problem:
- Comparator run and report:
- One primary change:
- Coupled changes (for example, crop changes Flatten/dense parameter count):
- Expected benefit and possible regression:
- Criteria for keeping the change (offline, driving, and cost):

## Reproduction

- Date and output directory:
- Exact command (only supported flags):
- Training Git commit, dirty state, and saved patch if applicable:
- Inference Git commit/dirty state if different:
- Python, Keras, backend, full dependency lock identity, hardware:
- Seed(s), determinism limitations:
- Dataset config, manifest locations and hashes; raw-data identity/exclusions:
- Counts by split; confirm unchanged validation membership or explain differences:
- Camera policy; correction value, sign, units, and out-of-range handling if used:
- Input shape, decoder/color order, dtype, normalization and location:
- Crop/resize/interpolation; augmentation settings/probability and training-only scope:
- Architecture/layer configuration, parameter count, regularization:
- Optimizer, learning rate, batch-size meaning, max epochs, patience:
- Source examples, augmented examples, and optimizer steps per epoch:
- Loss and selection metric:

Attach or link generated config; do not assume it captures fields the code does not save.

## Verification and training

- Relevant checks/tests, execution date, actual results:
- Smoke result if performed (not quality evidence):
- Training duration and hardware utilization if measured:
- Actual epochs completed; selected epoch (state zero- or one-based numbering):
- Checkpoint paths and SHA-256 hashes:
- Export/reload checks and scope:
- History and loss plot:

## Offline results

| Metric | Comparator | This run |
| --- | --- | --- |
| Overall MSE / MAE | | |
| Always-zero MSE | | |
| Left / near-straight / right MSE and counts | | |
| Per-session MSE | | |
| Inference latency / training duration | | |

- Metrics, predictions, scatter, and error-image paths:
- Interpretation and failure cases; do not diagnose solely from aggregate loss:
- Repeat seeds and variability, or state single-run evidence:

## Simulator attempts and video

- Simulator version, track, direction, start position:
- Speed target, controller/settings, steering scaling/clipping:
- Input preprocessing agreement verified:
- Predetermined completion/departure/intervention criteria:

| Attempt | Conditions / seed if applicable | Lap completion or progress | Departures / collisions / interventions / resets | Failure location and behavior | Video path/link and timestamps |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Record successful and failed attempts. Record video type (simulator view or camera
frames), duration, frame rate/timing, and durable location/hash. A camera-frame
video alone may not show all driving criteria; include observations or telemetry.
Do not claim a measured distance if only a visual landmark is known.

## Decision and next action

- Keep / reject / inconclusive, with supporting evidence:
- Was improvement repeatable? What remains untested?
- Next single experiment or integration fix:
- Update the experiment index and project-plan checkpoint.
