# STORM

Structured Trajectory generatiOn for Robot Motion planning.

STORM is a research workspace for generative, constraint-aware trajectory generation. The project treats a plan as a structured trajectory object rather than an isolated path: components may be coupled across robot bodies, time, history, environment geometry, and topology.

## Problem

Given:

- task conditions such as start states, goals, and goal sequences
- environment observations and obstacle geometry
- robot structure or state representation
- optional history or memory, such as executed paths or previous plans

Generate:

- one or more structured trajectory candidates

Subject to:

- goal reaching
- collision avoidance
- smoothness and dynamic feasibility
- spatial coupling constraints
- history-dependent and topology-dependent constraints

The initial target task is tethered-robot trajectory generation, where the executed history approximates the tether topology. The framework remains general enough to keep whole-body mobile manipulation as a structured-trajectory instance rather than the main project identity.
