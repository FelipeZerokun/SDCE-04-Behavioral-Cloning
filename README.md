# Behavioral Cloning — Python

A PyTorch-based driving pipeline that learns steering commands from
recorded camera images and controls a vehicle in the Udacity simulator.
The project uses UV for reproducible Python environments and dependency
management.

This repository modernizes the Behavioral Cloning project from Udacity's
Self-Driving Car Engineer Nanodegree as a typed, tested Python package
with a command-line interface.

**Status:** Initial data collection. The modern training and inference
pipeline is not yet implemented.

## Features

## Training and driving pipeline

## Example output

## Requirements

## Installation

## Usage

### Collect training data

Launch the simulator and enter Training Mode. Practice driving before
starting a recording.

1. Choose a separate output directory for each recording session.
2. Position the car near the center of the road.
3. Start recording and drive smoothly for a few laps.
4. Stop recording before resetting or repositioning the vehicle.
5. Verify that the session contains `driving_log.csv` and an `IMG` folder.

Begin with a short test recording to verify the output before collecting
full sessions.

Collect center-road driving in both directions and separate recovery
demonstrations from the left and right sides. During recovery collection,
record the return to the center, rather than the deliberate departure
from it.

Keep recordings organized by session, for example:

    data/raw/track1_center_clockwise_001/
    data/raw/track1_center_counterclockwise_001/
    data/raw/track1_recovery_left_001/
    data/raw/track1_recovery_right_001/

Document each session's simulator version, track, direction, driving
purpose, and any mistakes.

Entire recording sessions will be assigned to training, validation, or
testing before augmentation. Horizontal flipping will be performed in
the training pipeline, with the steering sign inverted.

Raw recordings are excluded from Git.

### Validate a dataset

### Train a model

### Evaluate a model

### Drive in the simulator

## Configuration

## Development

## Project structure

## Limitations

## Legacy project and attribution

## License