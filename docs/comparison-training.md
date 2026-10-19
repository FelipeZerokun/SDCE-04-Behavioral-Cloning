# Three-model training and driving comparison

## Eventlet diagnostic rollback (2026-10-19)

All three full training runs have now completed. Driving remains under evaluation.
At the user's request, drive.py uses Eventlet again for a backend comparison.
Eventlet 0.41.2 is installed and locked; Gevent dependencies remain available.
The direct model call, external normalization, target speed 9, and PI controller
are unchanged from the latest Gevent diagnostic. Eventlet's deprecation warning
is expected. This rollback does not yet establish the cause of repeated telemetry.

The original Eventlet recording contained 467 frames with one consecutive identical
pair; the first Gevent recording contained 484 frames, including 154 identical
triplets. These were different runs, not a controlled backend benchmark.
Logs now include backend/model/settings, connection IDs and active counts,
per-connection sequence and image-repeat counts, callback-entry intervals, speed,
steering, throttle, and inference/callback timings. Timings do not measure simulator
capture age or command delivery, and callback timing excludes the diagnostic print.
Use fresh output directories and preserve both successful and failed attempts.

Status as of 2026-10-19: all three full trainings and driving comparisons completed.
See [results and lessons](experiments/three-model-results.md). Commands below are
reproduction instructions; existing run directories must not be overwritten.
No new training is queued. The user reports a good video; its path and run metadata
are pending. Further data collection is a future hypothesis, not a proven improvement.

## Fixed comparison settings

| Setting | Value |
| --- | --- |
| Architectures | custom CNN (`baseline`), LeNet-style (`lenet`), NVIDIA-style (`nvidia`) |
| Training source records | 6,384, existing session assignments |
| Training camera views | 19,152: center, left, right |
| Camera correction | left +0.2, right -0.2; explicit clipping to [-1, 1], affecting 8 targets |
| Flipping | Each training camera view plus its horizontal mirror with negated steering |
| Training views / batches per full epoch | 38,304 / 1,197 at batch size 32 |
| Validation | 2,320 center images, no augmentation, existing session assignments |
| Input | Full 160 x 320 RGB float32, normalized externally by pixel / 255.0 - 0.5 |
| Geometry inside model | Crop top 50 / bottom 20; bilinear resize to 66 x 200 |
| Optimizer / learning rate | Adam / 0.001 |
| Seed | 42; does not guarantee bitwise determinism across environments |
| Training budget | At most 20 epochs, early-stopping patience 4 |
| Checkpoint selection | Lowest validation MSE |

Under this geometry the custom CNN has 100,361 parameters; LeNet-style has
1,186,361; NVIDIA-style has 1,595,511. The NVIDIA option follows the supplied
1164-wide dense-layer variant; it is not an exact reproduction of the original
paper's parameter count. See [architecture details](udacity-options.md).
Equal settings are a practical comparison, not individually optimized settings.
Early stopping can produce different training durations.

## Run sequentially from the repository root

Each output directory must be new. Do not include --smoke or --prepare-only for
these full runs. Do not run these commands merely to check documentation.

### Custom CNN

```powershell
.venv\Scripts\python.exe -u -m behavioral_cloning.train --architecture baseline --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --batch-size 32 --learning-rate 0.001 --seed 42 --epochs 20 --patience 4 --output-dir outputs/comparison-custom-01
```

### LeNet-style

```powershell
.venv\Scripts\python.exe -u -m behavioral_cloning.train --architecture lenet --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --batch-size 32 --learning-rate 0.001 --seed 42 --epochs 20 --patience 4 --output-dir outputs/comparison-lenet-01
```

### NVIDIA-style

```powershell
.venv\Scripts\python.exe -u -m behavioral_cloning.train --architecture nvidia --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --batch-size 32 --learning-rate 0.001 --seed 42 --epochs 20 --patience 4 --output-dir outputs/comparison-nvidia-01
```

Wait for each command to finish. Confirm the final saved-run message,
export_verification.json, and validation/metrics.json before starting the next.
Keep the terminal output and note elapsed training time; the full training command
does not currently save isolated fit duration or inference latency automatically.
Preserve failed/interrupted directories and use a new suffix for a fresh run.
There is no checkpoint-resume option in this CLI.

## Saved artifacts and interpretation

Each run saves config.json, architecture.json, source and lock snapshots, Git
identity/status/diff, original/effective manifests, data audits, history.csv,
loss.png, best model.keras and model.h5, checkpoint hashes, reload checks,
and validation metrics/predictions/error plots. The validation set is used for
checkpoint selection; it is not an independent final test set.

The smoke checks used only 64 training views and 16 validation images. Their
losses are pipeline diagnostics, not architecture rankings. Original baseline-01
used different geometry and training views, so it remains historical context,
not the controlled architecture comparator. Preserve it and its two failed drives.

## Driving comparison after training

Use simulator-windows-64, the initial track, normal direction and the same default
start for all models. Confirm and record the exact simulator build, start pose,
direction, speed units and conditions. Keep the original PI controller at target
9, Kp 0.1, Ki 0.002, throttle clipped to [-1, 1]. Restart the Python process between
attempts to reset its integral. Capture the simulator window, not only camera JPEGs.

For each model, use its model.h5 and a new driving directory, for example:

```powershell
.venv\Scripts\python.exe -m behavioral_cloning.drive outputs/comparison-custom-01/model.h5 outputs/comparison-custom-drive-01
```

Do not drive a smoke checkpoint as if it were the fully trained model. Attempt two
runs per full model under consistent conditions; record successes and failures,
collisions, interventions, resets and failure landmarks. One uninterrupted lap
requires no road departure, collision, manual correction or reset. Compare full
validation MSE/MAE, parameter counts, measured training time/inference latency,
and actual driving. Do not infer causality from one seed or small metric gaps.
Update the [experiment index](experiments/README.md) with actual completed runs.

## GitHub versus local artifacts

Git tracks code, documentation, configuration, tests and uv.lock. The raw data,
simulator, model files and recordings under outputs/ are intentionally ignored.
A GitHub push does not back up those artifacts. Keep separate durable copies and
record their locations/hashes before removing any local originals.
