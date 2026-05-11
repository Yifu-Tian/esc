# MM-Flow

MM-Flow generative replanning for mobile manipulators.

## Problem Statement

Given:

- current whole-body state
- previous receding-horizon plan
- recent executed state/action history
- local obstacle observation
- goal or task condition

Generate:

- the next receding-horizon whole-body trajectory

Subject to:

- differential-drive base constraints
- arm joint, velocity, and acceleration limits
- whole-body collision avoidance
- task progress or end-effector tracking
- temporal consistency with previous execution

## Repository Layout

- `src/`: core implementation.
- `configs/`: experiment and model configuration files.
- `experiments/`: runnable experiment entry points.
- `models/`: trained model checkpoints and model definitions when needed.
- `data/`: generated datasets and cached environments.
- `scripts/`: utility scripts.
- `notebooks/`: exploratory analysis.
- `docs/`: project notes and design documents.
- `papers/`: related-paper notes.
- `tests/`: focused tests for geometry, dynamics, and planners.
