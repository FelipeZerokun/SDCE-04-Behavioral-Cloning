# baseline-01: center-camera CNN

Status: full training and offline evaluation complete; two failed autonomous attempts.
Updated 2026-10-18 from existing artifacts and operator reports; no retraining.

## Purpose and configuration

Establish a manageable baseline using recorded steering and full center images
before trying crop, augmentation, side-camera corrections, or architecture variants.
No direct numeric comparison with the historical model is available.

Evidence directory: `outputs/baseline-01/` (local, Git-ignored). Sources are
`config.json`, `history.csv`, `validation/metrics.json`, and
`export_verification.json`. Training date, exact launch command, training Git
revision/dirty state, hardware, and duration were not captured in these inspected
records; do not substitute today's code revision as historical provenance.

| Setting | Recorded value |
| --- | --- |
| Raw recordings / assignment config | `data/raw` / `configs/dataset.json` |
| Training / validation examples | 6,384 / 2,320 |
| Smoke | false |
| Seed | 42 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Maximum epochs / patience | 20 / 4 |
| Python / Keras / backend | 3.12.13 / 3.15.1 / TensorFlow |
| Input | 160 x 320 x 3 RGB float32 |
| Normalization | pixel / 255.0 - 0.5, outside model |
| Crop / resize / augmentation | None / none / none |

The baseline implementation uses Adam, MSE with MAE reporting, four stride-2 ReLU
convolutions (16/24/32/48 filters), dense ReLU layers 64/16, and one linear output.
Architecture details describe the inspected implementation; layer configuration is
also carried by the saved model, but historical source identity was not recorded.

## Observed results

Six epochs completed. Best validation loss occurred at **epoch 2** (CSV index 1).
The recorded `effective_epochs: 20` is the requested maximum, not the completed count.
Four subsequent epochs did not beat epoch 2, consistent with patience 4.

| Epoch (one-based) | Training MSE | Validation MSE |
| --- | ---: | ---: |
| 1 | 0.012359 | 0.003963 |
| 2 | 0.006266 | 0.002234 |
| 3 | 0.004770 | 0.002816 |
| 4 | 0.003760 | 0.002568 |
| 5 | 0.002704 | 0.002444 |
| 6 | 0.002283 | 0.002396 |

Best-checkpoint offline evaluation:

| Group | Count | Model MSE | Always-zero MSE | Model MAE |
| --- | ---: | ---: | ---: | ---: |
| Overall | 2,320 | 0.002233803 | 0.007885963 | 0.032706942 |
| inverse_lap_02 | 1,201 | 0.001837336 | 0.007469756 | 0.028246238 |
| normal_lap_02 | 1,119 | 0.002659323 | 0.008332669 | 0.037494525 |
| Left | 442 | 0.005107236 | 0.020548468 | 0.056816749 |
| Near straight | 1,437 | 0.000778261 | 0.000569694 | 0.020950837 |
| Right | 441 | 0.004096743 | 0.019034830 | 0.046849771 |

Near straight means absolute target <= 0.05. Predictions were not clipped.
The model beats always-zero overall and for turns, but not within the near-straight
subset. This invites inspection of straight-driving behavior; it does not prove
oscillation or any other simulator failure. Training loss continued falling after
the best validation epoch; additional training did not improve the selected metric.
These are single-run findings on the validation set used for checkpoint selection.

## Artifacts and verification

- `dataset/`: run-specific audit and split manifests.
- `config.json`: arguments, versions, input contract, selected session/record IDs.
- `history.csv`, `loss.png`: training history.
- `model.keras`, `model.h5`: best checkpoint and compatibility export.
- `export_verification.json`: both reload checks passed on 32 examples.
- `validation/`: metrics, predictions CSV, scatter, largest-error images.

The existing training documentation reports 18 passing tests on 2026-10-09.
No tests were rerun for this documentation update. Reload agreement in the current
environment does not verify the legacy simulator script or an older Keras runtime.

## Simulator integration (2026-10-18)

The user copied `references/udacity/drive.py` into
`src/behavioral_cloning/drive.py` and adapted it. The historical reference is
unchanged. The active script now:

- Decodes camera images with PIL, explicitly converts to RGB float32, checks
  shape (160, 320, 3), and applies pixel / 255.0 - 0.5 exactly once.
- Extracts steering from prediction[0, 0]; no steering scaling or clipping.
- Loads model.h5 with compile=False for inference. Normalizes bytes/text in the
  Keras metadata comparison and closes the HDF5 metadata handle automatically.
- Uses timezone-aware UTC timestamps and refuses to reuse an existing image folder.
- Clips the original PI controller's throttle output to [-1, 1].

Attempt 01 used Eventlet. Its deprecation warning prompted migration to Gevent
with gevent-websocket, explicitly selected via async_mode="gevent", a Socket.IO
WSGIApp, and WebSocketHandler on 127.0.0.1:4567. Attempt 02 verified the live
connection after migration. Eventlet was removed from project dependencies.
The installed configuration retained python-socketio 4.6.1 and python-engineio
3.13.2 for simulator compatibility; project requirements specify gevent >=26.9.0
and gevent-websocket >=0.10.1. See uv.lock for the current resolved environment.
The old WebSocket dependency remains a limitation; this is not a fully modern stack.
Asyncio migration was considered and deferred; the user chose to retain Gevent.

Checks performed during integration: H5 inference loading succeeded; script syntax
passed; for one real validation image, the PIL/base64 inference input exactly
matched the existing OpenCV training preprocessor and yielded finite steering.
This is a limited spot check, not comprehensive pipeline validation. Original
training artifacts and model weights were not changed. Exact inference commit,
source snapshot, and dependency lock identity were not saved per attempt.

## Driving record and decision

Conditions and evidence:

- Simulator: local simulator-windows-64 on Windows; manual and autonomous modes
  worked. Exact simulator release/build hash was not captured.
- Track: initial track. Normal direction/default start were the instructed setup;
  precise start pose and direction were not independently recorded per attempt.
- Model: outputs/baseline-01/model.h5, unchanged baseline.
- Controller: target 9 simulator speed units; Kp=0.1, Ki=0.002, integral summed
  per telemetry event, output clipped to [-1, 1]. Units were not independently
  verified. Integral windup and update-rate dependence remain possible limitations.
- Failure criterion: leaving the drivable road. Success requires an uninterrupted
  full lap without departures, collisions, manual intervention, or resets.
- Video: no simulator-window video supplied or confirmed. Saved JPEGs are camera
  frames, not a completed video. Speed/steering were printed, not saved alongside
  frames. Collisions, interventions, resets, and exact departure time were not logged.

| Attempt (2026-10-18) | Backend | Operator-observed result | Saved evidence |
| --- | --- | --- | --- |
| 01 | Eventlet | Accelerated to approximately 9; initially near center, then drifted left. Tried to follow the first left-hand curve but departed to the right. No lap completed. | outputs/baseline-01-drive-01: 467 JPEGs |
| 02 | Gevent | Connection and driving worked; operator reported the same first-curve failure. No lap completed. | outputs/baseline-01-drive-02: 484 JPEGs |

Verified file inventories on 2026-10-18: attempt 01 filenames span
14:31:05.808 to 14:32:25.659 UTC; attempt 02 spans 14:58:36.092 to
14:59:56.785 UTC. These are recording spans, not lap times or departure times.
Frame contents have not yet been reviewed. Local outputs are Git-ignored and
are not a durable backup; no external video/artifact link has been supplied.

Reproduction commands used (each directory must be new):

```powershell
.venv\Scripts\python.exe -m behavioral_cloning.drive outputs/baseline-01/model.h5 outputs/baseline-01-drive-01
.venv\Scripts\python.exe -m behavioral_cloning.drive outputs/baseline-01/model.h5 outputs/baseline-01-drive-02
```

**Decision:** preserve the baseline and both failed attempts. Similar behavior across
backends suggests the failure is not specific to Eventlet, but does not establish
whether steering prediction, latency, speed control, or recovery is responsible.
No model-quality improvement is claimed from the transport migration.

**Next:** inspect the approach/curve/departure frames and recording gaps before
choosing one model experiment. Predictions can be recomputed on saved JPEGs, but
re-encoding can change them slightly; actual speed history and live commands cannot
be recovered exactly. Compare relevant training/validation examples, then state one
hypothesis and its tradeoffs. Cropping, flipping, targeted data, side cameras, and
architecture changes remain candidates, not approved training jobs.
