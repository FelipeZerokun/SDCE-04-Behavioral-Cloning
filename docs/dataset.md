# Dataset preparation

The 2026-10-08 milestone uses individual center-camera images and recorded
steering targets. No CNN is implemented. Collection history and session purposes
are in the [README](../README.md). The recorder confirmed visual quality with no
exclusions. Raw recordings remain unchanged.

## Reproduce the dataset

From the repository root:

```powershell
uv run python -m behavioral_cloning.data_validation
```

Optional arguments are `--raw-dir`, `--config`, and `--output-dir`.
Defaults are `data/raw`, `configs/dataset.json`, and `outputs/dataset`.
The output directory must be outside the raw recording directory.

The configuration assigns each folder exactly once. Training uses the `01` laps
and targeted recordings; validation uses the `02` laps. Neither shuffling nor
random seeds affect membership. The command preserves configuration order and
CSV row order, producing identical manifests from identical inputs.

The audit checks seven fields per row, finite steering within the simulator's
normalized command range [-1, 1], and readable uint8 center images with shape
(160, 320, 3). These steering values are commands, not degrees or radians.
It records invalid auxiliary telemetry as warnings because throttle, brake,
and speed are not model inputs or targets. Side images are not audited.

Every issue identifies a session and one-based CSV record; record 0 indicates
a session/file issue. CSV record numbers match the viewer's displayed frame
numbers for these headerless files. Missing labels are never imputed.

`report.json` records counts, shapes, issues, steering distributions, configuration,
unassigned folders, and exact image-file duplicates across splits. Such duplicates
block publication, as do required-data errors. Byte hashes cannot detect every
visually similar or re-encoded image; independent session collection remains the
main protection against neighboring-frame leakage.

On success, `train.csv` and `validation.csv` contain session, record, center path,
steering, and image SHA-256. Center paths are relative to the raw directory.
Manifests reference originals; they do not copy or transform images. Generated
outputs are Git-ignored. The command returns exit code 1 on a failed audit and
removes previous generated manifests before auditing, preventing their reuse
after a data-check failure. Configuration errors raise an exception before audit;
always require a successful command before consuming any existing output.

Exclusions are currently empty. If later justified, add an entry to the config's
`exclusions` list with `session`, `start_record`, `end_record`, and a nonempty
`reason`. Ranges are one-based and inclusive. They remain separate from raw data.
An excluded record is not audited or included in distributions/manifests.

## Verified results

All 8,704 records passed: no exclusions, errors, warnings, unassigned sessions,
or exact image-file duplicates across splits. All center images were 160 x 320
with three uint8 channels.

| Session | Split | Examples |
| --- | --- | ---: |
| normal_lap_01 | Training | 1,210 |
| inverse_lap_01 | Training | 1,170 |
| recover_left | Training | 371 |
| recover_right | Training | 401 |
| sharp_curve_left | Training | 866 |
| sharp_curve_right | Training | 803 |
| slight_curve_left | Training | 421 |
| slight_curve_right | Training | 411 |
| bridge | Training | 731 |
| normal_lap_02 | Validation | 1,119 |
| inverse_lap_02 | Validation | 1,201 |

Near straight is defined for reporting as absolute steering <= 0.05. It does not
change any label or determine exclusions.

| Statistic | Training | Validation |
| --- | ---: | ---: |
| Examples | 6,384 | 2,320 |
| Exact zero | 1,149 (18.0%) | 236 (10.2%) |
| Near straight | 3,486 (54.6%) | 1,437 (61.9%) |
| Steering < -0.05 | 1,453 | 442 |
| Steering > 0.05 | 1,445 | 441 |
| Minimum | -0.6176471 | -0.3411765 |
| Maximum | 1.0 | 0.3882353 |
| Mean | -0.0014130 | -0.0001572 |
| Always-zero predictor MSE | 0.0165881 | 0.0078860 |

Near-straight examples are the majority, although left/right commands are closely
balanced. Validation contains no commands beyond +/-0.5; training contains 62.
Retain the natural distributions initially. Do not rebalance validation. Any
later sampling or augmentation must be confined to training.

The zero predictor gives a reference for future MSE: a small error alone can be
misleading on mostly gentle steering. Compare errors by steering range as well as
overall MSE. This holdout measures new recordings of the same track, not unseen
tracks or dedicated recovery performance. Autonomous driving still needs testing.

## Baseline preprocessing contract

Use this conservative starting plan when implementing the future Keras pipeline:

| Property | Decision |
| --- | --- |
| Camera | Center only |
| Source image | Height 160, width 320, three uint8 channels |
| Cropping | None for the first baseline; retain the full field of view |
| Resizing | None; retain native resolution |
| Color order | RGB; convert OpenCV BGR once, leave RGB decoders in RGB |
| Model input | Channels-last float32, shape (batch, 160, 320, 3) |
| Normalization | float32 pixel / 255.0 - 0.5, applied exactly once |
| Target | float32 recorded steering, unchanged, one scalar per image |
| Augmentation | None for the first baseline; later training-only experiments |

Keeping native framing avoids committing to a crop before evaluating its effect
on road visibility. Cropping and downsampling remain possible later experiments;
the full image costs more computation and includes irrelevant scenery/hood.
The preprocessing function itself will be implemented with the training pipeline.
Training and simulator inference must call the same preparation logic, with the
decoder's input color order explicit. No additional normalization should then
occur inside the model. Persist this contract alongside each future model.

## Verification

Seven tests passed, covering the original loader plus malformed rows, missing and
nonfinite labels, corrupt/missing images, range errors, auxiliary warnings,
exclusions, empty sessions, split overlap, deterministic outputs, duplicate
detection, and distribution boundaries. Ruff and strict mypy passed for the new
validation module. These checks do not imply a model has been trained or evaluated.
