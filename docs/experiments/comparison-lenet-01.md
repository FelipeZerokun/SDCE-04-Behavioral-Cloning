# Experiment: comparison-lenet-01

Status: trained and driving-tested; reviewed 2026-10-19.

See [shared results, driving attempts and lessons](three-model-results.md) and
[reproduction configuration](../comparison-training.md).

- Output directory: outputs/comparison-lenet-01 (preserve).
- Training Git HEAD: 2563a2c3a5914975b5ee1a4d67c22e29ade24bab; saved working tree clean.
- Source and dependency hashes: provenance.json; snapshots under source/.
- Five epochs completed; selected epoch 1 (one-based), minimum validation MSE.
- Parameter count: 1186361; seed 42; single training run.
- Both model formats passed saved reload checks on 32 examples.
- Configuration: config.json; dataset identities: dataset/train-effective.json and dataset/validation-effective.json.
- History: history.csv and loss.png; evaluation: validation/metrics.json and accompanying predictions/plots.
- Training hardware and whole-run duration not independently recorded here.
- Driving code changed after training; Eventlet rollback and direct inference are described in the shared report. Current dirty working tree must not be represented as the training commit.

## Checkpoint SHA-256 hashes

```json
{
  "model.keras": "0683ac62cdf3b5c7f471b76168b60600390c7e8d741f7638667d0ca4ed5b8ce4",
  "model.h5": "c190c0e1efa4f6207bd6c6b721cfdaa3915d50c4078dd95f2cb7b63d04f91f60"
}
```

## Saved offline evaluation

```json
{
  "overall": {
    "count": 2320,
    "mse": 0.008019113055552625,
    "mae": 0.06958131729368107,
    "zero_mse": 0.007885962628408474
  },
  "by_session": {
    "inverse_lap_02": {
      "count": 1201,
      "mse": 0.008329507759101414,
      "mae": 0.07032210291452526,
      "zero_mse": 0.007469755832825736
    },
    "normal_lap_02": {
      "count": 1119,
      "mse": 0.007685972716891232,
      "mae": 0.06878624711438361,
      "zero_mse": 0.008332668938949019
    }
  },
  "by_steering": {
    "left": {
      "count": 442,
      "mse": 0.011892798140755109,
      "mae": 0.08192440473747348,
      "zero_mse": 0.020548467740140474
    },
    "near_straight": {
      "count": 1437,
      "mse": 0.005090313648894596,
      "mae": 0.05941967326965764,
      "zero_mse": 0.0005696941279655631
    },
    "right": {
      "count": 441,
      "mse": 0.013680146932441713,
      "mae": 0.09032199260516735,
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
