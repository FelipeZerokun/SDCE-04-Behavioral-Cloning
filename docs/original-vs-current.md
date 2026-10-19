# Original solution and current pipeline

This guide records the user's pasted original implementation and description of
the course progression, compared with the current repository inspected on
2026-10-13. Course outcomes are reported context, not reproduced experiments here.
Use it alongside the [project plan](project-plan.md), one topic at a time.

## Implementation update (2026-10-18)

The historical comparison below describes baseline-01. Optional horizontal flips,
audited side cameras, model-embedded crop/resize, and the supplied-diagram
NVIDIA-style variant are now implemented. The baseline defaults are unchanged.
All three full training and simulator comparisons have now completed. See the
[2026-10-19 results](experiments/three-model-results.md): custom has reported lap
completion; NVIDIA has best validation metrics but tight-curve failures persist.
Current inference uses Eventlet again after backend diagnostics. See [Udacity-style options](udacity-options.md) and
the [verification record](experiments/udacity-options.md) for current capabilities.

## Original learning progression

The order below follows the user's clarified history. "Improvement" identifies
the intended experiment, not a guarantee that every change improved driving.

0. **Initial baseline:** a simple Keras model using only center images, without
   preprocessing or normalization.
1. **Data preprocessing:** introduce preprocessing, including normalization.
2. **LeNet:** try a LeNet-style convolutional architecture.
3. **Image flipping:** mirror images and negate steering to address directional imbalance.
4. **Multiple cameras:** add left/right views with estimated recovery targets. The
   lecture also showed worse driving after this change: more examples can hurt.
5. **Cropping:** remove sky and hood to focus the input and reduce downstream computation.
6. **NVIDIA architecture:** try an NVIDIA-inspired network; inspect increasing validation loss and reduce
   training duration. Architecture capacity alone did not guarantee better driving.
7. **More data collection:** collect meaningful recovery and curve demonstrations
   and drive both directions.

Loss visualization and generators supported this process. The final student task
was to train/test configurations and record a full autonomous lap. Not every
intermediate implementation or measured result was supplied.

The final pasted code includes normalization, cropping, three cameras, mirrored
samples, a five-convolution network, dropout, a generator, and H5 saving. Its
implementation differs from the architecture described in the lecture, notably
in convolution strides. The present project preserves the incremental experimental
method, not an assumption that the final old architecture is the correct destination.

## Our approach compared with that progression

Our first trained baseline starts with several lessons from the original project
already included. It is a reference point for future experiments, not a recreation
of the original unnormalized simple network. We have not run an ablation removing
each initial choice, so we cannot assign a measured gain to each one.

| Concern | Original progression | Our approach before baseline-01 | Remaining experiment |
| --- | --- | --- | --- |
| Data coverage | Add meaningful data late in the progression | Collect both driving directions, actual recovery demonstrations, curves, and bridge passes before training | Collect more only when coverage/quality evidence warrants it |
| Pixel preprocessing | Add after the simple model | Explicit RGB float32 normalization from the beginning | Crop/resize separately if useful; preserve inference consistency |
| Architecture | Simple model -> LeNet -> NVIDIA-inspired network | Start directly with a small custom strided CNN | Change capacity, regularization, or architecture based on evidence |
| Directional coverage | Add horizontal flipping | Use real recordings in both directions; no flipping | Compare training-only flipping if it addresses observed weakness |
| Recovery | Add side cameras with estimated corrections, later more demonstrations | Use center images from actual recovery sessions with recorded steering | Side cameras remain optional, with audited images and justified corrections |
| Field of view | Add cropping | Keep full 160 x 320 images for the reference run | Inspect candidate crop on varied scenes, then compare a crop-only run |
| Evaluation | Inspect losses and iterate in simulator | Save offline metrics, best checkpoint, and reload checks from the first run | Two failed driving attempts now documented; complete lap/video still missing |
| Engineering | Much of the solution in one script | Modules, typed records, integrity checks, fixed session splits, tests, run artifacts | Copied drive.py adapted; Gevent comparison followed by Eventlet rollback; see current results |

Recorded recovery images contain driver actions from actual off-center positions.
Side-camera augmentation instead estimates the action appropriate to a displaced
view. Both-direction recording broadens real experience; mirroring produces derived
examples and does not replace independent collection. Neither strategy is proven
superior by our current single baseline run.

## Exactly what was our baseline?

**baseline-01** uses center-camera images from 6,384 training examples and 2,320
validation examples, with entire sessions kept in their assigned split. It includes
the recovery/curve/bridge training sessions. Inputs are full-size normalized RGB;
there is no crop, resize, augmentation, side-camera correction, or dropout.

The network is a **small custom Keras CNN**, named `steering_baseline`, not LeNet
and not a reproduction of the NVIDIA architecture:

| Layer | Configuration |
| --- | --- |
| Input | 160 x 320 x 3, float32, already normalized |
| Convolution 1 | 16 filters, 5 x 5, stride 2, ReLU |
| Convolution 2 | 24 filters, 5 x 5, stride 2, ReLU |
| Convolution 3 | 32 filters, 3 x 3, stride 2, ReLU |
| Convolution 4 | 48 filters, 3 x 3, stride 2, ReLU |
| Flatten | Convert spatial features to one vector per example |
| Dense layers | 64 ReLU -> 16 ReLU -> 1 linear steering value |

Convolutions use default valid padding. The network has about 475,000 parameters.
Strides reduce spatial dimensions and keep the dense part manageable. This is a
modest starting architecture; its adequacy for driving has not yet been established.

Training uses Adam at 0.001, batch size 32, seed 42, MSE loss, MAE reporting, up to
20 epochs, and early-stopping patience 4. Six epochs completed; epoch 2 supplied
the best checkpoint. Validation MSE is 0.002233803 versus 0.007885963 for always-zero
steering. See the [baseline report](experiments/baseline-01.md) for subgroup results
and provenance limits. No baseline autonomous lap or video has been established.

## How we improve from here

1. Understand and document this baseline, preserving its artifacts.
2. Verify simulator integration with matching preprocessing; drive and record the
   existing model under documented conditions. This is an integration milestone,
   not a model change or a reason to retrain.
3. Identify a concrete weakness from curves, predictions, and driving evidence.
4. Choose one experiment: crop, flip, targeted data, camera expansion, or a defined
   model/optimizer change. There is no mandatory ladder through all techniques.
5. Record its hypothesis/configuration, test the implementation, train into a new
   directory, evaluate the same validation data and driving conditions, and preserve
   both successes and failures. Repeat small gains before treating them as reliable.
6. Keep demonstrated benefits, test combinations explicitly, and work toward a
   documented uninterrupted autonomous lap. Do not promise each experiment will help.

The [project plan](project-plan.md) contains decision criteria for each candidate;
the [experiment index](experiments/README.md) records actual runs. This comparison
records why choices differ, while experiment reports record whether they worked.

## Classification

**Essential** means required for correctness or the stated evaluation goal.
**Safeguard** makes mistakes easier to detect. **Optional** means a design or
modeling choice whose benefit depends on context. Several approaches can satisfy
the same essential requirement; modularity itself does not improve predictions.

## 1. Loading and representation

- Original: pandas CSV -> NumPy metadata array -> separate paths and steering lists.
  Center/left/right lists are extended in matching order, preserving alignment.
- Current: raw records can be represented by `DrivingSample` in `dataset.py`;
  training consumes manifests as `TrainingExample` records in `training_data.py`.
  The audit reads raw CSV directly and reuses the path resolver; training does not
  call `load_session()`. Image pixels are decoded later for each batch.
- Reason/classification: image-label pairing is essential; named immutable records
  are a safeguard and readability choice. Metadata need not be NumPy just because
  the eventual model consumes numeric batches. Mixed strings/numbers typically
  produce an object array with little numeric-array benefit.
- Tradeoff/transfer: separate lists or tuples are adequate for a small notebook
  if alignment is maintained. Choose a representation that preserves relationships
  during filtering, reordering, and debugging. Type annotations do not validate values.
- Paths: both implementations relocate filenames into the local IMG folder. The
  old split handles backslashes; current `PurePosixPath` handles forward slashes.
  Neither handles both recorded styles universally. Current support matches the
  recorded Ubuntu paths; it is not a general portability fix.

## 2. Validation and preparation

- Original: assumes usable CSV fields, labels, and images; decoding happens later.
- Current: `prepare_dataset(raw_dir, config_path, output_dir)` reads selected sessions,
  audits integrity, reports findings, and publishes train/validation manifests on success.
  `dataset.json` assigns sessions and optional reasoned record exclusions; it does
  not configure CNN layers or learning rate. An audit is a systematic check; a
  manifest is an inventory of image paths, labels, and source identities, not pixels.
- Execution map: protect raw/output separation -> read config -> check nonempty
  split lists and unique plain session names -> build exclusion sets -> remove stale
  manifests -> call `validate_session()` -> check cross-split file hashes -> write
  report -> write manifests only on a passing audit -> return the pass boolean.
- Checks include seven CSV fields, finite steering in [-1, 1], decodable uint8 center
  images of shape (160, 320, 3), and valid exclusions. Auxiliary telemetry problems
  are warnings; required data problems block publication. Side images are not audited.
- Classification: usable aligned inputs are essential; an up-front audit and saved
  report are safeguards with extra I/O cost. A tiny trusted dataset can use simpler
  checks at load time. Neither method assesses whether steering demonstrations are good.
- Limitation: early configuration errors and some I/O failures raise exceptions;
  the docstring's "Always write diagnostics" is not an unconditional guarantee.
  A failed audit clears stale manifests, but errors before that stage may leave them.
  Consumers must require successful preparation. Stored hashes are not recomputed
  by `load_manifest()`; they do not guard later raw-file changes automatically.

## 3. Splitting

- Original: random 80/20 split after combining all three cameras. The `X_test` set
  is passed as validation during training, so its role is validation, not a final test.
- Current: entire recording sessions assigned in `configs/dataset.json`; independent
  laps validate, other laps and targeted demonstrations train. Checks reject session,
  path, or stored file-hash overlap.
- Reason/classification: separation appropriate to the intended generalization
  question is essential. Neighboring frames and synchronized cameras can otherwise
  cross splits, yielding optimistic validation. Session grouping is a safeguard
  suited to this dataset, not a universal splitting rule.
- Tradeoff/transfer: group splits can differ in size/distribution; exact 80/20 is not
  required. Random splitting can be sufficient for genuinely independent samples.
  Byte hashes miss similar or re-encoded images. These sessions cover the same track,
  and no independent final test set currently exists.

## 4. Preprocessing and augmentation

- Original: decoder supplies images; model normalizes using x/255 - 0.5 and removes
  50 top/20 bottom rows. Generator mirrors every image and negates its label, including
  validation. Side cameras use corrected labels (+0.5 left, -0.5 right in the code).
- Current: checked uint8 BGR input -> RGB -> float32 -> pixel/255 - 0.5, outside the
  model. Full 160 x 320 framing, no resizing, augmentation, or side-camera targets.
- Classification: consistent training/inference color, range, and shape are essential.
  Normalization location, crop, camera expansion, and augmentation are choices.
  Center-only retains observed labels and simplifies the baseline; it may lack some
  views that side cameras could contribute. Both-direction and recovery recordings
  address coverage through real demonstrations.
- Tradeoff/transfer: in-model preprocessing travels with the model; external functions
  are easy to test but callers must apply them correctly. Cropping can discard road
  cues. CPU slicing can be cheap; GPU location alone does not prove faster cropping.
  Mirroring preserves steering symmetry; camera corrections estimate missing actions.
  Decoder dtype/range must be checked before dividing by 255 (some formats decode
  to already-scaled floats). Validation augmentation changes what is measured; use
  unaugmented validation for this project's natural-image comparison.

## 5. Batching

- Original: infinite generator, slice paths, load images, add mirrored examples,
  build NumPy arrays, shuffle pairs, and yield. Intended to avoid loading all pixels.
- Current: `DrivingDataset` supplies indexed batches, reports batch count with ceil,
  shuffles training indices per epoch, keeps the final partial batch, and emits
  float32 labels shaped (batch, 1). Validation order stays fixed.
- Classification: correct coverage and alignment are essential; an indexed Keras
  dataset is optional. A correctly written generator is sufficient and memory-efficient.
  Both still hold metadata in memory. Current batch behavior has focused tests.
- Transfer: define whether batch size counts source samples or augmented images,
  and whether an epoch visits each source once. Test batch boundaries, not just shape.

## 6. Architecture

- Original pasted model: five ReLU convolutions, same padding, default stride 1;
  Flatten; dense widths 1164/100/50/10/1 with dropout between hidden dense layers.
  Hidden dense layers have no explicit nonlinear activations. Crop output is 90 x 320;
  the flattened convolution output is 1,843,200 values, making the first dense layer
  about 2.15 billion weights plus biases. This follows the pasted dimensions; it is
  not a measured historical model summary.
- Current: four stride-2 ReLU convolutions (16/24/32/48), Flatten, ReLU dense 64/16,
  linear scalar output; about 475,000 parameters, no dropout.
- Reason/classification: a scalar regression output is required here; width, depth,
  stride, activation layout, and regularization are design choices. Downsampling
  makes this baseline tractable. Calling a model NVIDIA-inspired does not verify
  its stride/layout matches the intended architecture.
- Tradeoff/transfer: more capacity costs memory, time, and possible overfitting;
  too little can underfit. Inspect tensor sizes and parameter counts before training.
  Missing hidden dense activations are a design limitation, not a syntax/runtime bug;
  the original convolution stack still contains nonlinearities.

## 7. Training

- Original: Adam/MSE, two epochs in the pasted final code, manual generator step
  counts, final in-memory model saved. Lecture described other durations while iterating.
- Current: explicit learning rate, seed, smoke option, inferred batch counts, MSE
  with MAE, best validation checkpoint, early stopping, nonfinite checks, CSV history.
- Classification: correctly specified steps/targets are essential. Seeding, callbacks,
  logging, and smoke runs are safeguards; exact optimizer and patience are choices.
- Tradeoff/transfer: a fixed small epoch count can be adequate for exploration.
  Early stopping uses validation and can stop amid noise. Falling training loss with
  worse validation is a clue, not proof of one cause. High losses can also indicate
  broken inputs or optimization, not just insufficient capacity.

## 8. Evaluation

- Original: course process inspects loss and simulator driving; the pasted script
  itself has no saved offline breakdown or driving evaluation implementation.
- Current: best-checkpoint predictions, MSE/MAE, zero predictor comparison, session
  and steering bins, scatterplot, and largest-error images. The copied drive.py supports live inference, currently via Eventlet after a Gevent
  comparison. Original baseline attempts failed; the later custom CNN completed
  reported laps. See the current results report.
- Classification: evaluation matching the goal is essential. Detailed diagnostics
  are safeguards; exact plots and bin thresholds are choices.
- Tradeoff/transfer: aggregate MSE can hide turn errors or reward near-zero predictions.
  Offline frames cannot show accumulating control errors, latency, or recovery from
  model-created positions. Demonstrate simulator laps separately and retain failures.

## 9. Saving and organization

- Original: saves final model to `model.h5`; much of the experimental context lives
  in one script or notebook state.
- Current: new directory per run, saved config/manifests, best `.keras` checkpoint,
  `.h5` export, reload prediction agreement on a fixed batch, history and metrics.
- Classification: preserving the tested model and its input requirements is essential;
  reload checks are safeguards; dual formats and module boundaries are optional.
- Tradeoff/transfer: H5 export does not guarantee old runtime compatibility, and a
  fixed-batch agreement check is not exhaustive. Current config lacks automatic Git
  revision, full experiment narrative, and driving/video records. The
  [report template](experiments/template.md) fills documentation gaps without pretending
  those fields are already generated by code.

## Actual defects and conditional issues in the pasted original

1. **Batch indexing bug:** the loop slices `batch_samples` but reads
   `X_samples[batch_sample]` and `y_samples[batch_sample]` from index zero each time.
   Later offsets repeatedly use the beginning of the split instead of the slice.
2. **Step-count mismatch:** up to 32 source samples produce 64 augmented images,
   but training requests ceil(2N/32) steps instead of ceil(N/32) for a source pass.
   Validation requests ceil(N) batches, not a count divided by batch size. This
   repeats data rather than providing the intended one-pass epoch/evaluation.
3. **Missing name in the supplied snippet:** `sklearn.utils.shuffle` references
   `sklearn` without binding that name via `import sklearn`; importing
   `train_test_split` alone does not do so. Earlier notebook imports could mask it.
4. **Conditional CSV bug:** default pandas header inference drops the first sample
   if the original log is headerless. It is correct if that CSV has a header.
5. **Conditional path bug:** backslash splitting fails for forward-slash absolute
   recordings. Current path handling has the opposite format limitation.
6. **Comment discrepancy:** comment says correction 0.2, executed setting is 0.5.

Random frame splitting is an evaluation weakness for correlated data, not a Python
execution bug. Cropping, dataclasses, dropout, camera count, and normalization
location should not be classified as bugs just because the current approach differs.

## Learning bookmark

Discussed: metadata versus numeric batches, dataclass purpose, recorded/local paths,
camera corrections, augmentation versus real collection, cropping, recovery data,
and the overall meaning of audits, manifests, and dataset configuration.

Next: read `prepare_dataset()` slowly from config loading onward. Explain each
comprehension, set operation, and nested call with a small example when encountered.
Do not treat this reference document as proof that every section has been taught.
