# Udacity-style pipeline capabilities

Status: implemented and preparation-verified; subsequent architecture smoke checks passed.
Full comparison training and driving evaluation remain pending.
Date: 2026-10-18. Comparator: [baseline-01](baseline-01.md).

This is an implementation record, not a training experiment result. User requested
horizontal augmentation, multiple cameras, cropping, and the supplied NVIDIA-style
network. All are optional and baseline defaults are preserved. See
[options and collection guide](../udacity-options.md) for exact semantics,
architecture differences, sample counts, provenance and supported commands.

Hypotheses: flipping may help directional imbalance; displaced camera views may
help recovery; crop may focus visual learning; the larger network may learn more
complex relationships. Risks: estimated or saturated side targets, lost crop cues,
more optimizer steps, overfitting and latency. None is a demonstrated improvement.

At the initial preparation milestone no trained checkpoint or driving result existed.
Subsequently, three authorized one-epoch smoke runs produced checkpoints and
16-image subset metrics; see the [index](README.md). They do not establish model quality.
The full-combination preflight is only a data/configuration check. Additional data
collection still requires the user in the simulator. A new training run requires
approval and a new output directory. No first model experiment has been selected.

## Verification results (2026-10-18)

- 27 tests passed, one existing synthetic-learning test deliberately deselected
  to avoid any training. Command: `.venv/Scripts/python.exe -m pytest -q -k
  "not model_learning_and_export"` (the command is one line).
- Tests cover mirrored pixel/label consistency and coverage, side-camera signs,
  missing/invalid images, source integrity, side-image hash overlap, explicit
  saturation, crop boundaries, NVIDIA feature-map shapes, baseline layer count,
  RGB/BGR preprocessing agreement, prevention of double normalization, native/H5
  reload agreement, and prepare-only configuration/provenance with fit forbidden.
- Ruff and strict mypy passed on the six changed training/support modules;
  Ruff also passed on the new tests. This is not a claim that the historical
  driving script or the entire repository passes strict typing.
- The first sandboxed test invocation hit a Windows/OpenCV invalid-handle
  diagnostic and stalled; it was stopped. The tests passed outside the sandbox.
  Upstream Keras/NumPy deprecation warnings remain during serialization tests.

Real-data preparation command (completed, no fit):

```powershell
.venv\Scripts\python.exe -m behavioral_cloning.train --prepare-only --architecture nvidia --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --output-dir outputs/udacity-options-preflight-01
```

| Item | Result |
| --- | ---: |
| Source training records | 6,384 |
| Camera-expanded training views | 19,152 |
| Training views including mirrors | 38,304 |
| Batches per epoch at batch size 32 | 1,197 |
| Center-only validation examples | 2,320 |
| Clipped side-camera targets at correction 0.2 | 8 |
| NVIDIA-style parameters | 1,595,511 |
| Unchanged baseline parameters | 475,145 |

Center and side-camera audits passed. The preflight validation CSV is byte-for-byte
identical to baseline-01's validation CSV. The new output directory contains
configuration, architecture, effective-example lists, audit and provenance; it
contains no trained checkpoint. The combined preflight is not an experiment result
and does not demonstrate an improvement. The 1164-wide dense layer explains much
of the increased parameter count versus the original paper caption.

Next decision: inspect failed baseline driving evidence and choose either an
isolated option comparison or an explicitly labeled combined Udacity-style run.
Further training needs separate user approval; additional raw data still needs collection.
