# ESC: Every Step Counts

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
