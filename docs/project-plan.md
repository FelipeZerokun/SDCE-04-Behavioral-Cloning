# Learning and experiment plan

Updated 2026-10-18. Goal: understand the implementation well enough to design
similar projects independently, and demonstrate one complete autonomous lap in
the simulator with a documented model, conditions, results, and video.
This goal concerns this simulator and track; it does not establish real-world autonomy.

## Where we are and what comes next

- [x] Collect and review center driving, both directions, recovery, curve, and bridge data.
- [x] Implement modular loading, auditing, session splits, preprocessing, batching,
  a small CNN, training, checkpoint export, and offline evaluation.
- [x] Complete full baseline training: [baseline-01](experiments/baseline-01.md).
- [x] Record the original approach and current differences in the
  [comparison guide](original-vs-current.md).
- [ ] **Current learning step:** slowly review `prepare_dataset()` in
  `data_validation.py`, starting with reading and checking `dataset.json`.
- [ ] Review baseline curves, predictions, and large errors together.
- [x] Adapt the copied Udacity drive.py and establish live inference (limited preprocessing spot check).
- [x] Replace deprecated Eventlet with Gevent and verify the live connection.
- [x] Preserve camera frames and operator reports from two failed baseline attempts.
- [ ] Inspect the first left-curve departure; distinguish pipeline/controller issues from model weakness.
- [ ] Capture simulator-window video and complete per-attempt conditions/telemetry.
- [x] Implement optional flip, audited side cameras, saved crop/resize, and NVIDIA-style architecture.
- [x] Select the three-model comparison: custom CNN, LeNet-style, NVIDIA-style under shared settings.
- [x] Complete the three authorized one-epoch smoke checks and record their outcomes (all passed).
- [ ] User to run the three full trainings sequentially using [the runbook](comparison-training.md), then evaluate offline and in the simulator.
- [ ] Record a complete autonomous lap and document the final model and limitations.

The baseline is already trained; do not retrain or replace it just to start this
plan. Documentation and inference integration can be added around it. Proposed
experiments below are hypotheses, not implemented features or proven improvements.

## Guided review agreement

Follow `train.py` in execution order and enter helper modules at their call sites.
Explain one small section, then pause for questions. For each section cover the
original intent, current behavior, reason for change, necessity, tradeoffs, and
how to reuse the reasoning. Explain unfamiliar syntax with small examples.
Distinguish actual bugs from preferences; newer code is not automatically better.
Do not train, install dependencies, or change implementation merely to explain it.

Review sequence:

1. Dataset preparation: config, audit, exclusions, split manifests, success/failure.
2. Manifest loading: `TrainingExample`, paths, labels, source identity.
3. Split checks: related frames, cameras, sessions, and duplicate files.
4. Seeding and smoke subsets; reproducibility limits.
5. Batching and preprocessing: follow a batch into image loading, RGB, normalization.
6. Saved configuration and its distinction from dataset configuration.
7. CNN architecture, shapes, parameters, and regression output.
8. Training, loss history, checkpoint selection, and stopping.
9. Reloading, exports, verification, and offline evaluation.
10. Simulator integration and driving evidence.

See the comparison guide's topic sections for reference; they are not a requirement
to teach the whole project in one session. Resume at the first unchecked learning
step rather than restarting the explanation.

## Baseline driving milestone

Current status: two attempts on simulator-windows-64 departed to the right during
the first left-hand curve. Gevent is retained; no model changes or new training
are selected. See the [baseline report](experiments/baseline-01.md) for the migration,
conditions, evidence limitations, and next inspection step. The learning bookmark
above belongs to the separate code-review chat.

The legacy `references/udacity/drive.py` passes raw RGB pixels to the model.
The current model requires normalized float32 RGB inputs. The copied active drive.py now applies the same normalization exactly once and respects decoder color order;
the existing preprocessing function accepts BGR, not RGB. File extension `.h5`
alone does not establish compatibility with the legacy script or dependencies.

Before judging model quality, verify model loading, image shape/color/range,
finite scalar prediction, steering sign, telemetry/control exchange, and frame
recording. Add focused tests for training/inference preprocessing agreement and
the adapter's inputs/outputs. A transport or preprocessing failure is not evidence
that the network needs more layers. No new training is required for this milestone.

Choose and record simulator version, track, direction, start position, and a fixed
speed/controller configuration before comparing runs. Define a departure as leaving
the drivable road; record collisions, manual interventions, resets, and completion.
Initial success: one continuous full lap without those failures, with video evidence.
Repeat under the same conditions to assess consistency. Other directions, speeds,
recovery starts, and the advanced track are separate evaluations, not implied successes.

## Experiment method

1. Inspect errors and driving video; rule out implementation/inference defects.
2. Write one hypothesis and one primary change in the [report template](experiments/template.md).
3. Identify the comparison run. Keep dataset membership, evaluation, and driving
   conditions fixed unless data or driving conditions are the explicit experiment.
4. Implement only the needed option, preserving baseline behavior. Test meaningful
   properties: crop boundaries, flip/label consistency, batch coverage, or reload agreement.
5. Use a small smoke run when needed to check the changed pipeline, then a full run
   in a new output directory. A smoke result is not model-quality evidence.
6. Record offline metrics and driving attempts, video, costs, and failure locations.
7. Keep, reject, or mark inconclusive. Preserve failed runs. Repeat promising small
   gains with additional seeds; seeds alone do not guarantee deterministic results.
8. Update the [experiment index](experiments/README.md) and this plan's next action.

Change one main factor at a time. After isolated experiments, useful combinations
can be tested explicitly. A lower validation MSE does not necessarily improve driving.
Track training time, parameter count, and inference latency as well as prediction errors.

Validation stays unaugmented and keeps its natural distribution. Repeated tuning
uses validation for development, so it is not a final independent test. No test set
exists yet; reserve fresh independent data/conditions before making broader claims.
If the dataset changes, record its new identity and, when needed, rerun the comparator
on the same data to distinguish a data effect from a model effect.

## Candidate experiments, selected by evidence

| Candidate | Question and trigger | Main tradeoff / required checks |
| --- | --- | --- |
| Crop | Does removing sky/hood improve efficiency or driving? A reasonable first controlled experiment after baseline driving if no stronger failure points elsewhere. | Inspect straight, curve, bridge, hill, and recovery frames. Original crop removes 50 top/20 bottom rows. It may remove anticipation cues. With Flatten, changing spatial size also changes dense parameter count; record this coupled effect. |
| Horizontal flip | Does mirrored training help directional weakness? | Flip the image and negate steering together, training only. Both-direction recordings already exist; extra benefit is unknown. Specify random probability versus duplicating every sample and resulting optimizer steps. |
| Targeted collection | Are actual failure situations absent or poorly demonstrated? | Record desired recovery/curve actions, not intentional departures. More redundant frames may add little. Keep validation independent. |
| Brightness variation | Is performance sensitive to appearance changes? | Preserve geometry/labels, use plausible ranges, training only. Not justified merely because augmentation is common. |
| Resize | Can lower resolution reduce compute without losing road cues? | Compare visibility and driving. Record interpolation, shapes, and parameter-count changes separately from cropping. |
| Learning rate / training duration | Does optimization fail to fit useful behavior? | Inspect curves first. Longer training is not automatically better; retain best-checkpoint selection. |
| Regularization | Does a repeatable training/validation gap suggest poor generalization? | Try a defined dropout or weight-decay change separately; excessive regularization can prevent learning. |
| Architecture | Does a verified small model lack useful capacity? | Try a documented modest change or NVIDIA-inspired variant. Record strides, activations, parameters, and latency; larger can overfit or become too slow. |
| Side cameras | Would additional off-center views improve recovery? | Audit side images, extend example metadata, split by source session before expansion, and define correction units/sign/range policy. Targets are estimates. Inference remains center-only; training-only camera expansion changes sample count and training budget. |

For side cameras, the original rule is left target = recorded steering + correction,
right target = recorded steering - correction (positive means right). A constant
correction ignores speed and geometry. Neither the lecture's 0.2 nor the pasted
code's 0.5 is established as appropriate here. Do not silently clip targets or
claim that three times as many views means three times as much independent evidence.

## Documentation and implementation discipline

Keep modules separated by responsibility, use useful type annotations, and add
runtime checks at data boundaries. Types do not enforce image shape or label quality.
Existing tests check data integrity, batch alignment, synthetic learning, exports,
and metric calculations; they cannot prove autonomous driving success.

Each implemented experiment must persist its actual preprocessing, augmentation,
architecture, training, dataset, and inference settings. The CLI now supports optional flip, camera expansion/corrections, crop/resize,
and architecture selection; see [Udacity-style options](udacity-options.md).
These are implemented capabilities, not completed training experiments. Keep raw data and baseline artifacts unchanged.

Commit small experiment reports and this index; store models/videos in separate
run directories or durable external storage with links and hashes. `outputs/` is
Git-ignored, so a local path alone is not a backup or a shareable artifact.
