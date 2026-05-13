# ESC: Every Step Counts

## Problem

Given:

```text
H: executed history trajectory
O: obstacle environment
x_t: current robot state
g: target goal
```

Generate a trajectory:

```text
tau = {x_t, ..., x_T}
```

such that:

```text
goal-reaching
collision-free
tangle-free / topology-consistent with history
smooth enough for execution
```

