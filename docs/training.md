# Single-image steering baseline

The training pipeline, smoke run, and full baseline-01 training are complete.
See the [baseline report](experiments/baseline-01.md) for measured results and the
[project plan](project-plan.md) for the learning sequence and next experiments.
Live inference is working; two autonomous attempts failed at the first left-hand
curve. Camera frames are saved; a complete lap and video remain unestablished.

For reproducing the completed three-model comparison, use the [sequential full-training
runbook](comparison-training.md). All three smoke checks passed; the user will
launch full runs manually. The examples below describe the default baseline.

## Run

From the repository root in PowerShell:

```powershell
uv sync --locked
uv run python -m behavioral_cloning.train --output-dir outputs/baseline-02
```

The example uses baseline-02 because baseline-01 already exists. Do not run it merely
to follow the documentation; the next step is inspecting the saved baseline driving failures.

Each run requires a new output directory to avoid overwriting previous models.
The command audits all recordings and creates fresh split manifests inside the
run directory. Audit failure stops training. It then trains for up to 20 epochs,
with batch size 32, Adam learning rate 0.001, seed 42, and early stopping after
four epochs without improved validation loss. The best validation checkpoint,
not necessarily the last epoch, is exported and evaluated.

For a short pipeline check:

```powershell
uv run python -m behavioral_cloning.train --smoke --batch-size 8 --output-dir outputs/smoke-02
```

Smoke mode audits the complete dataset but selects 32 training and 16 validation
examples reproducibly and runs one epoch. Its metrics are not a baseline result.

To evaluate the saved model again on its full held-out manifest:

```powershell
uv run python -m behavioral_cloning.evaluate outputs/baseline-01/model.keras --manifest outputs/baseline-01/dataset/validation.csv --output-dir outputs/baseline-01/reevaluation
```

The standalone evaluator can also load `model.h5`. It expects the same external
preprocessing contract as the training command. Evaluation output files are
replaced when reusing an evaluation directory.

## Module reference

For interactive learning, follow `train.py` in execution order as described in the
[project plan](project-plan.md). The list below is a module reference.

1. `preprocessing.py`: OpenCV decoding, BGR-to-RGB conversion, float32 conversion,
   and normalization by `pixel / 255.0 - 0.5`.
2. `training_data.py`: ordered manifest records and cross-split overlap checks.
3. `batches.py`: load one batch at a time; shuffle training only, keep the final
   partial batch, and return labels with shape `(batch, 1)`.
4. `model.py`: four strided convolutions (16/24/32/48 filters), flatten, dense
   layers (64/16), and one linear steering output. About 475,000 parameters.
5. `train.py`: audit, seed, train, select checkpoint, export, and evaluate.
6. `evaluate.py`: ordered predictions, metrics, and plots.

By default the model receives full 160 x 320 RGB images, already normalized,
without crop, resize or augmentation. Optional Udacity-style features are now
available; see [configuration and collection guide](udacity-options.md).
Crop/resize live inside saved models; training-only flips and side-camera
expansion never affect validation membership.
Targets are recorded steering commands, unchanged. Validation is never shuffled.
TensorFlow is the default Keras backend, using CPU on native Windows. GPU setup
is a separate environment decision. Dependencies and exact versions are locked
in `uv.lock`. Random seeds reduce variability but do not guarantee identical
results across hardware or backend versions.

## Artifacts

- `dataset/`: successful audit report and complete train/validation manifests.
- `config.json`: settings, selected sample identities, versions and preprocessing.
- `history.csv`, `loss.png`: epoch losses and validation curve.
- `model.keras`: best full native Keras checkpoint.
- `model.h5`: full-model compatibility export, without inference preprocessing.
- `export_verification.json`: prediction agreement after reloading both formats.
- `validation/metrics.json`: MSE, MAE, zero-predictor MSE, counts, by session and
  by left / near-straight / right steering; near-straight means abs(target) <= 0.05.
- `validation/predictions.csv`: sample identities, targets, and raw predictions.
- `validation/predictions.png`, `largest_errors.png`: prediction scatter and
  the six largest errors with source images.

Compare full validation MSE against the always-zero reference (~0.007886), and
inspect curves and turning errors. No separate test set is currently assigned.
Validation measures new recordings of this same track; it does not establish
recovery skill or autonomous driving success. Modern Keras H5 reload has been
checked; the historical Udacity script has not been validated with this export.

## Verification

```powershell
uv run python -m pytest -q
```

Tests cover preprocessing, manifest parsing and invalid data, split overlap,
batch alignment and final partial batches, model learning, native/H5 reload,
and evaluation metrics, in addition to the existing dataset audit tests.
On 2026-10-09, all 18 tests passed and the real-data smoke run completed,
including both model exports and evaluation plots. Keras emitted upstream
NumPy deprecation warnings during tests and its expected legacy-H5 warning.
