# ESC: Every Step Counts

**ESC** studies history-conditioned trajectory generation for tethered robot motion.

The core hypothesis is simple:

> For a tethered robot, future feasibility is not determined only by the current robot state, goal, and obstacles. Every executed step changes the topology memory of the tether, so the future trajectory must be generated conditioned on the full history.

In short:

```text
p(tau | x_t, goal, obstacles) is not enough
p(tau | history, x_t, goal, obstacles) is needed
```

## Problem

Given:

```text
H: executed history trajectory
O: obstacle environment
x_t: current robot state
g: target goal
```

Generate a future trajectory:

```text
tau = {x_t, ..., x_T}
```

such that it is:

```text
goal-reaching
collision-free
tangle-free / topology-consistent with history
smooth enough for execution
```

The initial focus is a 2D tethered point robot with circular obstacles. This is a controlled testbed for validating the central idea before moving to 2.5D / 3D scenes.

## Key Idea

The same start and goal may require different future paths depending on the past trajectory.

ESC treats the history as a structured memory rather than a passive context. A candidate future trajectory should be evaluated by its relation to:

```text
history segments
obstacle geometry
history-induced topology state
candidate-history coupling
```

This suggests a generative planning model of the form:

```text
tau ~ p_theta(tau | H, O, x_t, g)
```

and eventually:

```text
tau ~ p_theta(tau | relation_state(H, O), x_t, g)
```

## Current Research Direction

The first stage is to build a dataset generator and verifier:

```text
1. procedurally generate 2D obstacle scenes
2. generate history trajectories with topology memory
3. generate multiple future trajectory candidates
4. verify collision, goal reaching, winding change, and segment-level coupling
5. keep multiple feasible futures per condition
```

The resulting dataset should support learning a multi-modal conditional distribution over feasible future trajectories.

## Planned Models

Baseline models:

```text
CVAE
conditional diffusion
conditional flow matching
```

ESC-oriented models:

```text
history segment encoder
obstacle set encoder
relation graph / graph transformer
topology-aware trajectory generator
flow-field or potential-field decoder
```

The long-term goal is not just to sample many trajectories and filter them, but to make the model understand why a future trajectory is feasible with respect to history.

## Repository Layout

```text
esc/
  src/esc/          core library code
  experiments/      scripts for dataset generation, training, and evaluation
  tests/            unit tests
  data/             local datasets, ignored by git
  outputs/          generated results, ignored by git
  assets/           figures and media
```

## Short-Term Milestones

```text
M1: 2D manual case verifier
M2: segment-level candidate-history relation analyzer
M3: procedural dataset generator
M4: CVAE / diffusion / flow-matching baselines
M5: relation-aware generative model
M6: 2.5D / 3D extension
```

## Status

This workspace is initialized as a clean research workspace. Early prototypes and exploratory scripts are being migrated from the previous STORM workspace as the project narrows to ESC.
