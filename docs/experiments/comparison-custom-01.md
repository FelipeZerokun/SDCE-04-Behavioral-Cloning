# Experiment: comparison-custom-01

Status: trained and driving-tested; reviewed 2026-10-19.

See [shared results, driving attempts and lessons](three-model-results.md) and
[reproduction configuration](../comparison-training.md).

- Output directory: outputs/comparison-custom-01 (preserve).
- Training Git HEAD: ff37f7e7f319f8647428850e888330d2078185ab; saved working tree dirty; see source/git-status.txt and source/git-diff.patch.
- Source and dependency hashes: provenance.json; snapshots under source/.
- Five epochs completed; selected epoch 1 (one-based), minimum validation MSE.
- Parameter count: 100361; seed 42; single training run.
- Both model formats passed saved reload checks on 32 examples.
- Configuration: config.json; dataset identities: dataset/train-effective.json and dataset/validation-effective.json.
- History: history.csv and loss.png; evaluation: validation/metrics.json and accompanying predictions/plots.
- Training hardware and whole-run duration not independently recorded here.
- Driving code changed after training; Eventlet rollback and direct inference are described in the shared report. Current dirty working tree must not be represented as the training commit.

## Checkpoint SHA-256 hashes

```json
{
  "model.keras": "ae35f679a215787a626f5cc50362364862addb4358dd46e34f63b52c79dbdd94",
  "model.h5": "8759c06f70cdeaed34ab68c04c0100a04ecfd81cd630e14826b862576049d0df"
}
```

## Saved offline evaluation

```json
{
  "overall": {
    "count": 2320,
    "mse": 0.007726072627756903,
    "mae": 0.07159167461189181,
    "zero_mse": 0.007885962628408474
  },
  "by_session": {
    "inverse_lap_02": {
      "count": 1201,
      "mse": 0.007526382366073277,
      "mae": 0.07049388617232318,
      "zero_mse": 0.007469755832825736
    },
    "normal_lap_02": {
      "count": 1119,
      "mse": 0.00794039613471136,
      "mae": 0.0727699086743779,
      "zero_mse": 0.008332668938949019
    }
  },
  "by_steering": {
    "left": {
      "count": 442,
      "mse": 0.009315503005160729,
      "mae": 0.07590771994914097,
      "zero_mse": 0.020548467740140474
    },
    "near_straight": {
      "count": 1437,
      "mse": 0.005570942534871693,
      "mae": 0.06468692098881693,
      "zero_mse": 0.0005696941279655631
    },
    "right": {
      "count": 441,
      "mse": 0.013155536837878352,
      "mae": 0.08976500549011057,
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
