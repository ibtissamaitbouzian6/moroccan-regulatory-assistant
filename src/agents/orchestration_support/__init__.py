"""P3 support tools: pure, independent checks around the orchestrator (nothing here runs a graph).

- `state_validator`: invariants of the orchestrator's `GraphState`, as a list of issues.
- `plan_validator`: why a planner reply was or was not usable (diagnosis, no text kept).
- `orchestration_limits`: the limits of a request as one object, a deadline, and a config check.

Nothing in this package is imported by `src.agents.orchestrator`: it can be plugged in later,
from a test or from a node, without changing the existing graph. The dependency goes one way
only: `plan_validator` imports `graph.parse_json_or_none` so that its diagnosis follows
`plan_node` exactly, and `graph` imports nothing from here (no import cycle).
"""
