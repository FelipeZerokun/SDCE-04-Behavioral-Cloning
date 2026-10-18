# Experiment index

Follow the [project plan](../project-plan.md). Copy the [template](template.md)
for each new experiment; use the same ID for its report and output directory.
Record hypotheses before running and conclusions afterward. Never overwrite the
baseline or silently replace a failed result. Proposed experiments are not queued
training jobs and require no action until selected.

| Run | Main change | Validation MSE | Driving / video | Decision |
| --- | --- | ---: | --- | --- |
| [baseline-01](baseline-01.md) | Initial small center-only CNN; no crop or augmentation | 0.002233803 | Two reported first-curve departures; camera frames saved, video unconfirmed | Preserve; inspect failure evidence before a model experiment |
| [comparison-smoke-custom-01](comparison-smoke-custom-01.md) | One-epoch architecture smoke check | 0.004584662 (16-image smoke subset) | Not driven | Pipeline passed; not model-quality evidence |
| [comparison-smoke-lenet-01](comparison-smoke-lenet-01.md) | One-epoch architecture smoke check | 0.065128843 (16-image smoke subset) | Not driven | Pipeline passed; not model-quality evidence |
| [comparison-smoke-nvidia-01](comparison-smoke-nvidia-01.md) | One-epoch architecture smoke check | 0.005025503 (16-image smoke subset) | Not driven | Pipeline passed; not model-quality evidence |

Next: the user will launch full custom/LeNet/NVIDIA training manually, one at a time,
using the [shared-configuration runbook](../comparison-training.md). No full run
has been launched here. All three smoke checks passed. Smoke metrics are not
comparable to baseline-01's full validation result. Keep one row per actual
training run, including repeat seeds, and separate driving attempts under its report.

Reports are versioned documentation. Large models, raw data, and videos are local
or externally stored artifacts. Keep durable copies and record locations/hashes;
Git-ignored `outputs/` is not backed up by committing this index.

Implementation only: [Udacity-style options](udacity-options.md) are implemented and preparation-verified (27 non-training tests);
Three one-epoch smoke checks have now passed; no full comparison training or
demonstrated driving improvement is claimed.
