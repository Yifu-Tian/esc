# MM-Flow

MM-Flow generative replanning for mobile manipulators.

## Research Idea

This project studies receding-horizon whole-body trajectory generation for mobile manipulators. Instead of generating each plan from Gaussian noise or a static primitive, the planner uses the shifted previous whole-body plan, the executed state/action history, the current local observation, and the task goal as the source information for generating the next feasible trajectory.

The central question is:

> Can plan-to-plan generative replanning produce faster, safer, and more temporally consistent whole-body trajectories for mobile manipulators than noise-based generation or pure optimization warm-starts?

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

## Development Roadmap

1. Simplified 3D mobile manipulator
   - 3D base and 4-7 DoF arm.
   - Cuboid, cube, cylinder, and sphere obstacles.
   - Link-level collision checks and trajectory parameterization.

2. Full robot model
   - Import the target mobile manipulator URDF.
   - Use realistic geometry, point cloud observations, and closed-loop execution.
   - Evaluate dynamic obstacles, limited perception, latency, and sim-to-real behavior.

## Current Demo

Run the simplified 3D sequential reaching demo:

```bash
cd /home/yifu/mm-flow
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py
```

Optional arguments:

```bash
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py --seed 11 --obstacle-count 16
```

Use the older geometric scaffold if needed:

```bash
PYTHONPATH=src python3 experiments/three_d_reach_sequence_demo.py --planner heuristic
```

Outputs:

- `outputs/three_d_reach_sequence/reach_sequence_3d.png`
- `outputs/three_d_reach_sequence/reach_sequence_3d.npz`
- `outputs/three_d_reach_sequence/reach_sequence_3d.gif`
- `outputs/three_d_reach_sequence/reach_sequence_3d.html`

This demo uses a ground mobile base and a simplified 3-DoF spatial arm to reach a sequence of 3D goal points around randomly sampled obstacles. Obstacles can be grounded or floating and include spheres, cubes, cuboids, and vertical cylinders. The default planner is now whole-body RRT-Connect; the heuristic planner remains available only as a scaffold.

For an interactive Matplotlib window, run this from a graphical desktop session:

```bash
PYTHONPATH=src python3 scripts/view_three_d_matplotlib.py
```

Matplotlib interaction depends on the local GUI backend. The HTML viewer is usually more portable for drag/zoom playback.

## Initial Baselines

- Noise-to-trajectory diffusion or flow matching.
- Shifted previous trajectory with trajectory optimization only.
- Primitive-based trajectory generation.
- Classical local planner or MPC if available.

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
