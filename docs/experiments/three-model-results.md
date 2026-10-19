# Three-model results and simulator investigation

**Project state: pending.** Successful autonomous laps have been reported with the
custom CNN. Further data collection is needed for the next improvement phase,
including targeted tight-turn/recovery examples and planned advanced-track data.
Improvement from these additions remains to be tested. No new training is queued.

Updated 2026-10-19. Status: all three models trained and driving-tested; work paused
at the user's chosen checkpoint. No further training or data collection is queued.
Operator reports, file checks, and timing measurements are distinguished below.

## Shared training and offline results

See [reproduction commands and configuration](../comparison-training.md). All three
runs used identical effective training and validation manifests: 6,384 source
training records, 19,152 camera views, 38,304 views including horizontal mirrors,
and 2,320 unaugmented center validation images. All data came from the initial
track; advanced-track training data has NOT been collected or used. Existing data
already includes both directions, curves, bridge, and left/right recovery sessions.
External RGB float32 normalization is pixel / 255 - 0.5 exactly once. Crop 50/20
and resize 66x200 are embedded in the model. Side corrections are +/-0.2 with
explicit clipping affecting 8 labels. Adam 0.001, batch 32, seed 42, maximum 20
epochs, patience 4; architectures differ, shared settings are not individually tuned.

All completed five epochs and selected epoch 1 (one-based) by lowest validation
MSE. Training loss declined while validation MSE did not improve: consistent with
an increasing generalization gap, not proof of a pipeline bug. Final evaluation
uses the best checkpoint, not epoch 5. Native Keras and H5 reload checks passed
on 32 examples per model. The H5 legacy-format warning is not a failed export.

| Model | Parameters | Validation MSE | MAE |
| --- | ---: | ---: | ---: |
| Custom CNN | 100,361 | 0.007726073 | 0.071591675 |
| LeNet-style | 1,186,361 | 0.008019113 | 0.069581317 |
| NVIDIA-style | 1,595,511 | 0.006894699 | 0.065568368 |
| Always-zero steering | — | 0.007885963 | Not recorded here |

NVIDIA has the lowest aggregate errors, but custom has the strongest reported lap
completion. One seed per architecture and validation-based checkpoint selection
limit generalization claims. baseline-01 used different inputs/training views and
is historical context, not the controlled architecture comparator. The NVIDIA
variant follows the supplied 1164-unit dense-layer diagram, not an exact original
paper reproduction. Full training duration was not measured; final-epoch console
figures (custom 115s, LeNet 138s, NVIDIA 158s) are not whole-run durations.

## Driving conditions and observed outcomes

Simulator: simulator-windows-64; exact build/version identifier not recorded.
Requested comparison conditions: initial track, normal direction, default start,
target speed 9 (reported as mph), PI Kp=0.1, Ki=0.002, throttle clipped to [-1,1].
Per-attempt start pose, resets and interventions were not exhaustively logged.
The user explicitly confirmed no intervention during the custom model's two laps.
A lower-speed NVIDIA attempt used target 6; it did not resolve the failures.

| Local output folder / attempt | Backend | Operator report |
| --- | --- | --- |
| comparison-custom-drive-01 | Gevent | One attempt: two uninterrupted autonomous laps, considerable swerving and slow recovery; later departed and could not recover |
| comparison-lenet-drive-01 | Gevent | One attempt: swerving, failed on a left-hand curve before completing a lap |
| comparison-nvidia-drive-01 / -02 | Gevent | Two attempts: smoother centering; repeated unrecoverable departure at the same tight right-hand curve |
| NVIDIA reduced-speed attempt | Gevent | Target 6 did not fix tight-curve departures; exact folder association not conclusively recorded |
| comparison-nvidia-drive-03 | Gevent | Additional curve/recovery recording; repeated JPEGs verified; do not infer speed from folder name |
| comparison-nvidia-diagnostic-01 | Gevent, model.predict | Diagnostic excerpt near speed 9: departure/recovery, slow prediction and repeated incoming image hashes |
| comparison-nvidia-diagnostic-02 | Gevent, direct inference | Failure persisted; user confirmed collision caused near-zero speed while controller commanded full throttle |
| custom-eventlet-test-01 | Eventlet, direct inference | Again completed a full lap without crashing; tight-curve recovery imperfect |
| lenet-eventlet-test-01 | Eventlet, direct inference | Better subjectively, but departed at second tight curve (right-hand); operator observed steering reduce from about 6 to 3 degrees |
| nvidia-eventlet-test-01 | Eventlet, direct inference | Same general tight-curve failures despite smoother-looking recorded frame sequence |

The three supplied LeNet frames at 06:43:47.992, 06:43:49.060 and 06:43:50.780 UTC
show approach to the outside left edge of a right turn, followed by departure toward
the water. Images alone do not prove why steering changed. Earlier NVIDIA logs
reached normalized commands around -0.35 to -0.42: there is no demonstrated hard
limit at +/-3 degrees. The assumed normalized-to-25-degree mapping is not verified
against simulator source. Do not increase steering gain based solely on that assumption.

## Eventlet, Gevent and inference findings

The original adapted driver used Eventlet. It was migrated to Gevent because
Eventlet is deprecated; Gevent was a compatibility migration, not full asyncio
modernization. On 2026-10-19 the user requested an Eventlet rollback to investigate
repeated images. Eventlet 0.41.2 was installed with permission and added to the
lockfile; Gevent packages remain installed. Current driver explicitly selects
Eventlet, binds 127.0.0.1:4567, retains external normalization, and uses
model(batch, training=False).numpy(). Historical reference scripts are unchanged.
The expected Eventlet deprecation warning remains visible.

SHA-256 checks of consecutive saved files found:

| Original baseline recording | JPEGs | Identical consecutive run lengths |
| --- | ---: | --- |
| baseline-01-drive-01 (Eventlet) | 467 | 465 singles, one pair |
| baseline-01-drive-02 (Gevent) | 484 | 10 singles, 6 pairs, 154 triplets |
| comparison-nvidia-drive-03 (Gevent) | 1,629 | 46 singles, 22 pairs, 513 triplets |

Diagnostics subsequently confirmed repeated incoming image payload hashes, not
just repeated saved JPEGs. Repetition was strongly associated with these Gevent
runs, but its root cause was NOT isolated. Backend configuration, connection or
simulator behavior remain possible factors. Eventlet sequences looked continuous
to the user, but final driving outcomes remained broadly similar. Do not conclude
that Gevent is defective or that the backend explains the model ranking.

| Pasted diagnostic excerpt | model.predict | Direct inference |
| --- | ---: | ---: |
| Callbacks / image groups | 63 / 22 | 92 / 34 |
| Excerpt span | 11.733 s | 6.057 s |
| Mean inference | 166.40 ms | 42.95 ms |
| Mean callback | 171.89 ms | 48.37 ms |
| Mean changed-image interval at callback entry | 547.42 ms | 183.55 ms |

These are different live excerpts, not a controlled offline benchmark. The direct
call substantially reduced observed processing cost but did not fix driving.
Independent prediction-equivalence benchmarking was proposed but not completed.
Callbacks decode, infer, send and save once per message. Slow processing does not
itself create three identical payloads. Filenames are save timestamps, not capture
timestamps. Callback timing excludes printing, send-return timing does not confirm
simulator receipt, and actual image age/backlog was not measured. The PI integral
accumulates once per callback without elapsed time, so changed callback frequency
also changes its effective integral action; speed-controller timing remains a limitation.

Current logs print backend, model, target speed and controller settings; connection
IDs/active counts; per-connection sequence, consecutive image-repeat count and
callback-entry interval; speed, steering, throttle and processing times. A repeated
hash does not establish duplicate socket connections. No message-dropping workaround
or steering multiplier was implemented.

## Video and evidence status

The user reports having a good video of a model running. Exact video path, model,
attempt, capture type (simulator window or camera sequence), duration and timestamps
have not been supplied or inspected. Lap completion is supported by operator
reports; do not label the video as independently verified lap evidence yet.
Follow-up: associate the video with its checkpoint and attempt, record a durable
location and hash, and identify the uninterrupted lap segment. Saved JPEGs alone
are not simulator-window video. Local outputs and raw data are Git-ignored and
need separate backup. Preserve successful and failed recordings.

## Conclusions and return point

- Custom CNN is the working initial-track result: repeat lap completion reported,
  with swerving/recovery limitations and an eventual failure in the longer run.
- NVIDIA's better offline errors and smoother behavior did not ensure tight-curve
  reliability. Closed-loop errors change subsequent inputs; offline metrics alone
  cannot evaluate recovery or sustained turning.
- Faster inference was worthwhile but insufficient. Eventlet rollback improved
  perceived recording continuity without resolving the common driving failures.
- Stop here at the user's chosen checkpoint; no new training is authorized or queued.

When returning, first compare custom/NVIDIA predictions on identical pre-departure
frames and human-labelled tight-turn/recovery examples. This audit remains undone.
Then select one evidence-based data experiment. The user wants advanced-track data
and more off-track recovery demonstrations. Hypothesis: more relevant curves and
corrective actions may improve anticipation and recovery. Improvement is NOT
established: advanced-track appearance/geometry may shift the distribution, redundant
frames add little, and poor recovery labels can hurt. Additional initial-track
failure-location examples are a distinct intervention from adding a new track.

Collect smooth successful sharp turns and deliberate, safely recoverable offset
starts with corrective steering, not long uncontrolled departures. Split new
recordings by session before camera/flip expansion; retain independent validation
and evaluate both tracks separately. Preserve current checkpoints as comparators,
use new output directories, record data/code identities, ask before new training,
and compare under fixed driving conditions. Do not claim advanced-track capability
from initial-track success or assume a larger network/more data guarantees improvement.
