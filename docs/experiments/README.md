# Experiment index

**Project state: pending.** Successful autonomous laps have been reported with the
custom CNN. Further data collection is needed for the next improvement phase,
including targeted tight-turn/recovery examples and planned advanced-track data.
Improvement from these additions remains to be tested. No new training is queued.

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

All three full runs completed and were driven. See [combined results and backend
investigation](three-model-results.md). Work is paused at the user's chosen checkpoint.
A good video is reported available; model/run association and durable video location
remain to be documented. No further training is queued.

| Full run | Validation MSE | Operator-observed driving |
| --- | ---: | --- |
| [comparison-custom-01](comparison-custom-01.md) | 0.007726073 | Two laps under Gevent before failure; another full lap under Eventlet |
| [comparison-lenet-01](comparison-lenet-01.md) | 0.008019113 | Tight-curve departures under both backends |
| [comparison-nvidia-01](comparison-nvidia-01.md) | 0.006894699 | Smoother but repeated tight-curve failures; target 6 and faster inference did not resolve them |

Reports are versioned. Git-ignored outputs, models, data and video need separate
backup. Future work: saved-frame/label audit, then one selected data experiment;
advanced-track and more recovery data are proposals, not proven improvements.
