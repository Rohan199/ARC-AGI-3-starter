"""Tests for the agent's pure grid logic. No game engine, no framework.

Imports the functions out of agent/my_agent.py by source-extraction so the
agent file stays completely self-contained (it must, since only that one
file ships to Kaggle).

Run:  .venv/bin/python scripts/test_agent.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"
text = SRC.read_text()

# Take everything before the class definition: the pure helpers, minus the
# framework imports that only resolve inside the harness.
if "# --- BEGIN PURE ---" not in text:
    raise SystemExit("my_agent.py is missing the PURE markers.")
head = text.split("# --- BEGIN PURE ---")[1].split("# --- END PURE ---")[0]
head = "import hashlib\nfrom collections import Counter\n" \
       "from typing import Any, Optional\n" + head
ns: dict = {}
exec(compile(head, str(SRC), "exec"), ns)

settled_grid = ns["settled_grid"]
grid_key = ns["grid_key"]
detect_scale = ns["detect_scale"]
downsample = ns["downsample"]
background_colour = ns["background_colour"]
components = ns["components"]
click_candidates = ns["click_candidates"]

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n         got  {got!r}\n         want {want!r}")


def blocks(pattern, k):
    """Render a small pattern scaled up by k, like the games do."""
    out = []
    for row in pattern:
        big = []
        for v in row:
            big.extend([v] * k)
        for _ in range(k):
            out.append(list(big))
    return out


print("settled grid")
anim = [[[1, 1], [1, 1]], [[9, 9], [9, 9]], [[5, 5], [5, 5]]]
check("takes last grid", settled_grid(anim), [[5, 5], [5, 5]])
check("passes through single grid", settled_grid([[3, 3], [3, 3]]),
      [[3, 3], [3, 3]])
check("none safe", settled_grid(None), None)

print("\ngrid key")
check("same grid same key", grid_key([[1, 2]]), grid_key([[1, 2]]))
check("different grid different key",
      grid_key([[1, 2]]) == grid_key([[1, 3]]), False)

print("\nscale detection")
p = [[0, 1, 0], [1, 0, 1]]
check("detects 4x blocks", detect_scale(blocks(p, 4)), 4)
check("detects 2x blocks", detect_scale(blocks(p, 2)), 2)
check("unscaled is 1", detect_scale([[0, 1, 0], [1, 0, 1]]), 1)
# A grid that is uniform is trivially block-structured at the largest size.
check("uniform grid picks largest", detect_scale([[7] * 8 for _ in range(8)]), 8)

print("\ndownsample")
check("recovers pattern", downsample(blocks(p, 4), 4), p)

print("\nbackground")
check("most common colour", background_colour([[0, 0, 0], [0, 1, 2]]), 0)

print("\ncomponents")
small = [
    [0, 0, 0, 0],
    [0, 5, 0, 0],
    [0, 0, 0, 7],
    [0, 0, 0, 0],
]
comps = components(small, 0)
check("finds two objects", len(comps), 2)
check("centres correct", sorted((r, c) for r, c, _, _ in comps),
      [(1, 1), (2, 3)])

joined = [
    [0, 0, 0],
    [0, 4, 4],
    [0, 0, 0],
]
check("adjacent same colour is one object", len(components(joined, 0)), 1)

print("\nclick candidates")
grid = blocks(small, 4)
cands = click_candidates(grid)
check("returns pixel coords", all(0 <= x < 64 and 0 <= y < 64
                                 for x, y in cands), True)
check("includes object centres", (6, 6) in cands, True)
check("no duplicates", len(cands) == len(set(cands)), True)
# The whole point: far fewer than 4096 blind options.
check("candidate set is small", len(cands) < 40, True)

big_noise = [[(r * 3 + c) % 9 for c in range(64)] for r in range(64)]
check("cap respected on noisy grid", len(click_candidates(big_noise, 20)) <= 20,
      True)
check("empty grid safe", click_candidates([]), [])

# ---------------------------------------------------------------------
# Interface tests: these need the real arcengine enum. The pure-logic
# tests above deliberately strip framework imports, which is exactly why
# they could not catch GameAction(6) raising ValueError.
# ---------------------------------------------------------------------
print("\nframework interface")
try:
    from arcengine import GameAction
except ImportError:
    print("  skip (arcengine not importable here)")
else:
    by_id = {a.value: a for a in GameAction}
    check("every id resolves to a member",
          sorted(by_id) == [0, 1, 2, 3, 4, 5, 6, 7], True)
    check("id 6 is the complex action", by_id[6].is_complex(), True)
    # The trap: direct construction does NOT work on this enum.
    try:
        GameAction(6)
        direct_works = True
    except ValueError:
        direct_works = False
    check("direct GameAction(6) is unusable (documents the trap)",
          direct_works, False)
    agent_src = SRC.read_text()
    check("agent does not call GameAction(aid) directly",
          "GameAction(aid)" in agent_src, False)

print(f"\n{'=' * 40}\n  {PASS} passed, {FAIL} failed\n{'=' * 40}")
sys.exit(1 if FAIL else 0)
