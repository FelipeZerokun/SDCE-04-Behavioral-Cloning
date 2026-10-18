# comparison-smoke-nvidia-01

Status: smoke verification passed; full training and driving not performed.
Date: 2026-10-18. Architecture: nvidia.

## Purpose and comparison

Verify fitting, best-checkpoint saving, native/H5 reloads, and offline evaluation
before full training. This is a pipeline check, not a model-quality experiment.
Compare with the other two comparison-smoke runs under the same settings. The
original baseline-01 remains historical context: its input geometry and training
views differ. No simulator driving is planned for these smoke checkpoints.

Shared configuration: crop 50 top/20 bottom, bilinear resize 66 x 200 in the model;
external RGB float32 normalization; three training cameras with correction 0.2
and explicit target clipping; original plus horizontal mirror with negated labels.
Validation is center-only and unaugmented. Adam 0.001, batch size 32, seed 42.
Smoke selects 32 camera-expanded training examples, then mirrors to 64 views;
validation uses 16 selected examples. Actual budget: one epoch / two training
batches, regardless of --epochs 20. Full data/camera auditing precedes subsetting.

## Reproduction and provenance

```powershell
.venv/Scripts/python.exe -u -m behavioral_cloning.train --smoke --architecture nvidia --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --batch-size 32 --seed 42 --learning-rate 0.001 --epochs 20 --patience 4 --output-dir outputs/comparison-smoke-nvidia-01
```

Output: outputs/comparison-smoke-nvidia-01. Source snapshot, Git HEAD/status/diff, dependency lock,
dataset configuration and effective-example identities are saved by the command.
Console output and whole-command timing are saved beside the directory as
comparison-smoke-nvidia-01.console.log and comparison-smoke-nvidia-01.execution.json.
Seed equality alone does not guarantee identical numerical results across runtimes.

## Results and decision

Completed one epoch / two training steps. Both native Keras and H5 reload checks
passed on 16 held-out examples. Offline metrics/predictions and plots were saved.
Model SHA-256 hashes were recomputed and matched model-hashes.json. Full audit
manifests and effective smoke examples (including camera, label and hash) are
byte-for-byte identical across all three runs. No split membership was changed.

| Item | Value |
| --- | ---: |
| Parameters | 1,595,511 |
| Training MSE (epoch 1) | 0.062767342 |
| Smoke validation MSE (16 examples) | 0.005025503 |
| Smoke validation MAE | 0.047274054 |
| Always-zero MSE on this subset | 0.004930796 |
| Whole-command elapsed seconds | 44.38 |

Elapsed time includes imports, complete center/side-image audit, fitting, exports,
and evaluation. It is not isolated training time and should not be used to rank
compute efficiency (filesystem caching also differs between sequential runs).
The console log reports the short fit duration separately. Runtime: Python
3.12.13, Keras 3.15.1, tensorflow backend;
TensorFlow CPU execution on Windows. Saved dependency lock/source snapshot carries
further provenance. Expected HDF5 legacy-format and TensorFlow informational messages
occurred; the process exited successfully.

Checkpoint hashes:

- model.keras: bee312c298929a89070187737a7d76a8fd4044d8b682928347b78eeb17ac0ce1
- model.h5: 473ed2e22ea00086325af0a736a07b04c21652c41a489b8c513d77a6a8e2a18c

**Decision:** pipeline check passed. These losses are not reliable architecture
quality comparisons after only two optimizer steps. No model ranking, full-lap
claim, or simulator video is produced. Preserve the artifacts; full comparison
training needs separate authorization and three new directories.
