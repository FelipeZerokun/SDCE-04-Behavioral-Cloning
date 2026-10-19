# Experiment: comparison-nvidia-01

Status: trained and driving-tested; reviewed 2026-10-19.

See [shared results, driving attempts and lessons](three-model-results.md) and
[reproduction configuration](../comparison-training.md).

- Output directory: outputs/comparison-nvidia-01 (preserve).
- Training Git HEAD: 2563a2c3a5914975b5ee1a4d67c22e29ade24bab; saved working tree clean.
- Source and dependency hashes: provenance.json; snapshots under source/.
- Five epochs completed; selected epoch 1 (one-based), minimum validation MSE.
- Parameter count: 1595511; seed 42; single training run.
- Both model formats passed saved reload checks on 32 examples.
- Configuration: config.json; dataset identities: dataset/train-effective.json and dataset/validation-effective.json.
- History: history.csv and loss.png; evaluation: validation/metrics.json and accompanying predictions/plots.
- Training hardware and whole-run duration not independently recorded here.
- Driving code changed after training; Eventlet rollback and direct inference are described in the shared report. Current dirty working tree must not be represented as the training commit.

## Checkpoint SHA-256 hashes

```json
{
  "model.keras": "3e4e179536c16169ab339dee43fa38e44e991323978c7d4b81ed2c257224b0a5",
  "model.h5": "b60dd7bf5b95ab9a0d296cbea0e2fbc23957f79259e83bf91d8c996da01dcf17"
}
```

## Saved offline evaluation

```json
{
  "overall": {
    "count": 2320,
    "mse": 0.006894698619022689,
    "mae": 0.06556836817866961,
    "zero_mse": 0.007885962628408474
  },
  "by_session": {
    "inverse_lap_02": {
      "count": 1201,
      "mse": 0.007128656260250913,
      "mae": 0.06582054739760464,
      "zero_mse": 0.007469755832825736
    },
    "normal_lap_02": {
      "count": 1119,
      "mse": 0.006643596628750037,
      "mae": 0.06529770933868662,
      "zero_mse": 0.008332668938949019
    }
  },
  "by_steering": {
    "left": {
      "count": 442,
      "mse": 0.008937913165104028,
      "mae": 0.07312658143142421,
      "zero_mse": 0.020548467740140474
    },
    "near_straight": {
      "count": 1437,
      "mse": 0.0052192969977440575,
      "mae": 0.059168796591842154,
      "zero_mse": 0.0005696941279655631
    },
    "right": {
      "count": 441,
      "mse": 0.010306152814962465,
      "mae": 0.07884604190328075,
      "zero_mse": 0.019034830147118046
    }
  },
  "near_straight_threshold": 0.05,
  "prediction_clipping": false
}
```

## Decision

Preserve this checkpoint and all attempts. No further training launched.
See the shared report for operator-observed outcomes, incomplete video metadata,
backend findings and the next proposed data audit. Model weights and videos are
not backed up by Git.
