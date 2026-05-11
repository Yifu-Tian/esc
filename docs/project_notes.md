# Project Notes

## Name

MM-Flow: Plan-to-Plan Mobile Manipulator replanning.

## Core Distinction

The project is not only about conditioning on a previous action or completing sparse keyframes. The target problem is coupled whole-body trajectory generation:

```text
tau(t) = [
  x_base(t), y_base(t), theta_base(t),
  v_base(t), omega_base(t),
  q_arm(t), qdot_arm(t)
]
```

These variables share one time axis and jointly determine the end-effector pose, link poses, collision geometry, nonholonomic feasibility, smoothness, and task progress.

## Candidate Paper Framing

Potential title:

> Plan-to-Plan Flow Matching for Safe Receding-Horizon Whole-Body Replanning of Mobile Manipulators

Main contribution candidates:

1. History-informed source distribution using shifted previous whole-body plans.
2. Whole-body trajectory generation for coupled base-arm systems.
3. Safety-aware generation or lightweight projection using link-level constraints.
4. Closed-loop dynamic-obstacle evaluation under limited local perception.

## Minimal Experiment

Start with a planar robot:

```text
base: x, y, theta
arm: q1, q2
state: [x, y, theta, q1, q2]
```

Inputs:

- shifted previous trajectory
- current state
- goal
- 2D obstacle map

Output:

- next horizon trajectory

Metrics:

- success rate
- collision rate
- inference time
- smoothness
- temporal consistency
- number of refinement steps

