# Udacity-style training options

Implemented 2026-10-18. These are configurable pipeline capabilities, not measured
model improvements. baseline-01 and both driving attempts remain preserved.
Three one-epoch architecture smoke runs were subsequently authorized. Full training
remains a separate decision; see the [experiment index](experiments/README.md).

## Input and saved-model contract

All models accept (160, 320, 3) RGB float32, normalized externally with
pixel / 255.0 - 0.5. OpenCV training images and PIL simulator images now share
preprocess_rgb(). Never normalize twice. Cropping and optional bilinear resizing
are saved Keras layers: training, standalone evaluation, native/H5 reloads, and
the current drive.py all use the same geometry automatically. Do not crop again
in drive.py. Existing baseline models remain compatible.

## Options and tradeoffs

| Flag | Behavior | Tradeoff |
| --- | --- | --- |
| --flip | Each training image appears once unchanged and once horizontally mirrored, with negated steering | Doubles views and optimizer steps; assumes mirror-consistent labels |
| --cameras all --camera-correction VALUE | Expand each selected training record to center, left (+VALUE), right (-VALUE) | Triples views; correction is an estimated target, not measured recovery steering |
| --camera-target-policy reject/clip | Default rejects corrected labels outside [-1, 1]; explicit clip saturates and records the count | Saturated targets lose correction magnitude |
| --crop-top 50 --crop-bottom 20 | Remove 50 top and 20 bottom rows inside model | Removes sky/hood but may remove useful distant-road cues; also changes Flatten parameter count |
| --resize 66 200 | Resize after cropping, bilinear, inside model | Changes resolution and aspect ratio; can lose small visual cues |
| --architecture lenet | LeNet-style RGB regressor: Conv6/pool/Conv16/pool, Dense120/84/1 | Large flattened feature map; not an exact historical LeNet reproduction |
| --architecture nvidia | Supplied-diagram NVIDIA-style CNN; requires --resize 66 200 | Greater capacity and computation; no guarantee of better driving |
| --prepare-only | Audit data, build architecture, save configuration/provenance, then exit before fit | No learned model or model-quality result is produced |

Defaults preserve the original full-size center-only baseline with no flips or
spatial transforms. Each flag can be selected independently (NVIDIA requires
66 x 200 resizing). Validation always stays center-only and unaugmented; it uses
the same saved spatial transforms as training. Flipping is deterministic dataset
expansion, not a random probability. Batch size counts final views, including
mirrors. Each full epoch covers every view once, with a final partial batch.
A three-camera-plus-flip run has six times the original training view count.

## Three-architecture comparison

The user selected a bounded comparison with identical data, crop/resize, augmentation,
optimizer settings and evaluation conditions. Under crop 50/20 and resize 66 x 200:

| CLI architecture | Parameters | Description |
| --- | ---: | --- |
| baseline | 100,361 | Existing custom CNN with the shared spatial preprocessing |
| lenet | 1,186,361 | Conv6(5x5)/average pool2, Conv16(5x5)/average pool2, Flatten, Dense120/84/1 |
| nvidia | 1,595,511 | Supplied-diagram variant described below |

LeNet hidden layers use ReLU and the regression output is linear. All three
passed forward prediction and native/H5 reload checks without training before
the smoke runs. The original full-resolution baseline-01 has 475,145 parameters
and remains historical context rather than the controlled architecture comparator.

## NVIDIA-style architecture

After optional crop and required 66 x 200 resize:

- Valid Conv2D: 24 filters, 5 x 5, stride 2 -> 31 x 98.
- Valid Conv2D: 36 filters, 5 x 5, stride 2 -> 14 x 47.
- Valid Conv2D: 48 filters, 5 x 5, stride 2 -> 5 x 22.
- Valid Conv2D: 64 filters, 3 x 3, stride 1 -> 3 x 20.
- Valid Conv2D: 64 filters, 3 x 3, stride 1 -> 1 x 18.
- Flatten -> Dense 1164 -> 100 -> 50 -> 10 -> 1.

Hidden layers use ReLU; output is linear. No extra normalization or dropout.
The supplied image explicitly labels a 1164-neuron fully connected layer. We
implement it literally. The flattened convolution output has 1152 values;
adding Dense(1164) makes this larger than the approximately 250,000 weights
claimed by the figure caption. This is an NVIDIA-style Udacity variant, not an
exact reproduction of the original paper. Actual parameter count is 1,595,511 (baseline: 475,145), saved in
config.json. RGB and simulator steering units are retained for this project.
Reference: https://arxiv.org/abs/1604.07316.

## Camera audit and provenance

The original center-image audit and session split run first. Only training
sessions are expanded; exclusions and source record numbers remain effective.
Every selected side image is checked for decoding, shape/type, path containment,
and a SHA-256 hash. The source center path, steering, and hash must agree with
the audit manifest. Split-overlap checks run again after expansion, including
side-image hashes. Failures stop preparation/training and record camera-audit.json.

Camera correction must be explicitly supplied. 0.2 is an example, not a tuned
value. Current training data has eight records whose corrected targets exceed
[-1, 1] at 0.2; rejection is the default, so the combined example explicitly uses
clipping. That choice must be retained in experiment comparisons.

Each new output directory captures arguments, view counts, steps per epoch,
parameter count, architecture JSON, effective examples (camera/label/hash),
original split manifests, a dataset-config copy, source files, dependency lock,
Git HEAD/status/diff, and hashes. Completed training additionally writes checkpoint
hashes, metrics and reload checks. Historical baseline provenance is not invented.
Source snapshots include untracked modules. For smoke runs, effective-example
records describe the reduced subset; camera auditing still covers the full split.

## Preparation example (no training)

From the project root, use a new output directory:

```powershell
.venv\Scripts\python.exe -m behavioral_cloning.train --prepare-only --architecture nvidia --crop-top 50 --crop-bottom 20 --resize 66 200 --flip --cameras all --camera-correction 0.2 --camera-target-policy clip --output-dir outputs/udacity-options-preflight-02
```

This combined command verifies all capabilities; it is not an isolated scientific
comparison. Request approval before any training run, including --smoke. After
approval, select the intended flags, omit --prepare-only, and use another new
output directory. Never reuse a preflight or baseline output directory.
The existing evaluate and drive commands accept the resulting model.h5/model.keras
as supported by each command (drive currently uses H5 metadata and requires H5).

## Additional data collection

Collection is a simulator activity, not something augmentation can replace.
Inspect the failure sequence first. Candidate sessions: centered demonstrations
through the first left curve, transition away from colored curbs, and deliberate
recoveries toward center from both offsets. Record desired corrections, not
unlabeled road departures. Use new named sessions; retain raw originals.

Add new session names to the training list in a new dataset configuration and
pass --config to training/preparation. Keep existing validation sessions fixed
for comparison; reserve fresh full laps separately for independent checks.
Record simulator build, track, direction, input device, intent and exclusions.
Do not silently add new frames to baseline-01 or claim collection is complete.

## Experiment discipline

The baseline failed twice at the first left curve. Marking dependence is a
hypothesis, not established causality. These options do not establish a remedy.
First inspect recorded frames and failure timing. Select one primary experiment
and document its hypothesis, coupled effects, comparator and conditions before
training. Preserve all runs and compare offline metrics and actual driving.
If the user chooses all options together, label it a combined experiment and
avoid attributing any improvement to one feature.

Verification: 27 non-training tests passed, lint and strict typing passed on the
changed training/support modules, and full real-data preparation passed. See the
[implementation record](experiments/udacity-options.md) for counts and limitations.
