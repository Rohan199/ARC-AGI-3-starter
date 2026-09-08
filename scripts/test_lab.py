"""Self-tests for the Phase 1 library. No game engine required.

Run:  .venv/bin/python scripts/test_lab.py

These cover the pure logic -- hashing, level segmentation, baseline
resolution, scoring against the official calculator, and the decision
helpers. The parts that touch a live environment are exercised by actually
running probe.py; they cannot be meaningfully faked here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from lab import (
    AVO_FALLBACK_BASELINE,
    frame_layers,
    LevelRecord,
    Step,
    actions_for_target,
    cliff_actions,
    hash_frame,
    marginal_value,
    observed_action_space,
    resolve_baselines,
    score_run,
    segment_levels,
    waste_ratio,
)

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n         got  {got!r}\n         want {want!r}")


def close(name: str, got: float, want: float, tol: float = 1e-6) -> None:
    global PASS, FAIL
    if abs(got - want) <= tol:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: got {got!r} want {want!r}")


@dataclass
class FakeEnvInfo:
    baseline_actions: Optional[list] = None


def mkstep(idx, changed_hash, levels, avail=(1, 2, 3), state="NOT_FINISHED"):
    return Step(
        idx=idx, action="ACTION1", frame_hash=changed_hash, changed=False,
        state=state, levels_completed=levels, available_actions=list(avail),
    )


print("hashing")
check("stable across calls", hash_frame([[[1, 2], [3, 4]]]),
      hash_frame([[[1, 2], [3, 4]]]))
check("differs on change", hash_frame([[[1, 2]]]) == hash_frame([[[1, 3]]]), False)
check("none handled", hash_frame(None), "none")

# A frame is a LIST OF GRIDS. Two runs may differ in the intermediate
# animation grids while settling on the identical final state.
run_a = [[[1, 1]], [[9, 9]], [[5, 5]]]   # 3 grids, settles on [[5,5]]
run_b = [[[1, 1]], [[7, 7]], [[5, 5]]]   # different middle, same settle
check("settled hash ignores animation",
      hash_frame(run_a) == hash_frame(run_b), True)
check("full hash sees animation",
      hash_frame(run_a, settled=False) == hash_frame(run_b, settled=False),
      False)
check("layer count", frame_layers(run_a), 3)
diff_settle = [[[1, 1]], [[9, 9]], [[6, 6]]]
check("settled hash still catches real change",
      hash_frame(run_a) == hash_frame(diff_settle), False)

print("\nwaste ratio")
steps = [
    Step(0, None, "a", False, "NOT_FINISHED", 0),
    Step(1, "ACTION1", "a", False, "NOT_FINISHED", 0),   # no change
    Step(2, "ACTION2", "b", True, "NOT_FINISHED", 0),    # changed
    Step(3, "ACTION3", "b", False, "NOT_FINISHED", 0),   # no change
    Step(4, "ACTION4", "c", True, "NOT_FINISHED", 0),    # changed
]
close("half wasted", waste_ratio(steps), 0.5)
check("empty run safe", waste_ratio([]), 0.0)

print("\naction space discovery")
mixed = [mkstep(0, "a", 0, avail=(1, 2, 3)), mkstep(1, "b", 0, avail=(3, 7))]
check("union of available", observed_action_space(mixed), [1, 2, 3, 7])

print("\nlevel segmentation")
# 3 actions clear level 1, 2 more clear level 2, 4 more fail to clear level 3
seq = [mkstep(0, "s", 0)]
seq += [mkstep(1, "a", 0), mkstep(2, "b", 0), mkstep(3, "c", 1)]
seq += [mkstep(4, "d", 1), mkstep(5, "e", 2)]
seq += [mkstep(i, f"f{i}", 2) for i in range(6, 10)]
levels = segment_levels(seq, "NOT_FINISHED")
check("three levels found", len(levels), 3)
check("level 1 actions", levels[0], LevelRecord(1, True, 3))
check("level 2 actions", levels[1], LevelRecord(2, True, 2))
check("trailing incomplete", levels[2], LevelRecord(3, False, 4))

won = [mkstep(0, "s", 0), mkstep(1, "a", 0), mkstep(2, "b", 1)]
check("no phantom level on WIN", len(segment_levels(won, "WIN")), 1)

print("\nbaseline resolution")
b, est = resolve_baselines(FakeEnvInfo([10, 20, 30]), 3)
check("real baselines used", (b, est), ([10, 20, 30], False))
b, est = resolve_baselines(FakeEnvInfo(None), 2)
check("fallback flagged", (b, est),
      ([AVO_FALLBACK_BASELINE, AVO_FALLBACK_BASELINE], True))
b, est = resolve_baselines(FakeEnvInfo([10, 20]), 3)
check("partial extended + flagged", (b, est), ([10, 20, 15], True))

print("\nscoring (official calculator)")
# Matching baseline exactly -> 100 per level.
r = score_run("t1", [LevelRecord(1, True, 10)], FakeEnvInfo([10]), "WIN")
close("exact match scores 100", r.score, 100.0)
check("not estimated", r.estimated, False)

# Double the actions -> (1/2)^2 = 25.
r = score_run("t2", [LevelRecord(1, True, 20)], FakeEnvInfo([10]), "WIN")
close("2x actions -> 25", r.score, 25.0)

# 5x the actions -> the cliff, 4 points.
r = score_run("t3", [LevelRecord(1, True, 50)], FakeEnvInfo([10]), "WIN")
close("5x actions -> 4", r.score, 4.0)

# Beating the baseline caps the LEVEL at 115, but the completion cap holds
# the GAME at 100. The 115 headroom only lets a fast level offset a slow one
# inside the same game -- it can never push a game above 100.
r = score_run("t4", [LevelRecord(1, True, 1)], FakeEnvInfo([10]), "WIN")
close("level score caps at 115", r.level_scores[0], 115.0)
close("game score caps at 100", r.score, 100.0)

# The offsetting effect: a fast level pulls a slow one up.
r = score_run("t4b", [LevelRecord(1, True, 1), LevelRecord(2, True, 40)],
              FakeEnvInfo([10, 10]), "WIN")
check("fast level offsets slow one", r.level_scores, [115.0, 6.25])

# Weighting: level 2 counts double level 1. Perfect L1, failed L2.
# total = (100*1 + 0*2)/3 = 33.33, capped by max_weights (1/3)*100 = 33.33
r = score_run("t5", [LevelRecord(1, True, 10), LevelRecord(2, False, 99)],
              FakeEnvInfo([10, 10]), "NOT_FINISHED")
close("later levels weigh more", r.score, 100.0 / 3.0, tol=1e-4)

# Failing everything scores zero, not an error.
r = score_run("t6", [LevelRecord(1, False, 80)], FakeEnvInfo([10]),
              "NOT_FINISHED")
close("all failed -> 0", r.score, 0.0)

print("\ndecision helpers")
check("cliff at 5x", cliff_actions(36), 180)
check("actions for 100 == baseline", actions_for_target(36, 100.0), 36)
check("actions for 25 == 2x baseline", actions_for_target(36, 25.0), 72)
# Squared metric: savings are worth much more when already efficient.
near = marginal_value(36, 40)
far = marginal_value(36, 200)
check("marginal value higher when efficient", near > far, True)

print(f"\n{'=' * 40}\n  {PASS} passed, {FAIL} failed\n{'=' * 40}")
sys.exit(1 if FAIL else 0)
