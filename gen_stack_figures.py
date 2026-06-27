"""Generate Experiment K (Stack World) figures from results/stack.json."""

import json
import sys

from echo_civilization import visualization as V

src = sys.argv[1] if len(sys.argv) > 1 else "results/stack.json"
with open(src) as fh:
    payload = json.load(fh)

outs = [
    V.plot_stack_frontier(payload, "figures/stack_frontier.png"),
    V.plot_stack_reliability(payload, "figures/stack_reliability.png"),
    V.plot_stack_culture_growth(payload, "figures/stack_culture_growth.png"),
]
for o in outs:
    print("wrote", o)
