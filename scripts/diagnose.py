"""Phase 1 diagnostics.

NOTE the filename: do NOT call this inspect.py. Anything in scripts/ shadows
the stdlib module of the same name, and `dataclasses` imports `inspect`
internally, so the failure surfaces somewhere completely unrelated.

Two jobs:

  --baselines      Real per-level human baselines straight from the
                   environment metadata (NOT from probe logs, which are
                   truncated to however far your agent actually got).

  --divergence G   Re-run G twice and report which cells differ, separating
                   settled state from animation frames.

Usage:
    .venv/bin/python scripts/diagnose.py --baselines
    .venv/bin/python scripts/diagnose.py --divergence lf52
"""
from __future__ import annotations

import argparse
import collections

from lab import (
    build_script,
    build_scripted_agent_class,
    frame_layers,
    hash_frame,
    _settled,
)

import arc_agi
from arc_agi import OperationMode


def show_baselines() -> None:
    """Read baselines from EnvironmentInfo directly.

    The probe summaries only carry as many baselines as the agent had levels,
    so a run that clears nothing shows exactly one. This reads the source.
    """
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = sorted(arc.get_environments(), key=lambda e: e.game_id)

    print(f"{'game':6} {'lvls':>4} {'per-level human baseline actions':44} "
          f"{'total':>6} {'cliff':>6}")
    print("-" * 76)
    totals, counts, all_levels = [], [], []
    for e in envs:
        gid = e.game_id.split("-")[0]
        b = list(getattr(e, "baseline_actions", None) or [])
        if not b:
            print(f"{gid:6} {'?':>4} {'(no baseline_actions in metadata)':44}")
            continue
        shown = str(b[:9]) + ("..." if len(b) > 9 else "")
        print(f"{gid:6} {len(b):>4} {shown:44} {sum(b):>6} {sum(b)*5:>6}")
        totals.append(sum(b))
        counts.append(len(b))
        all_levels.extend(b)

    if not totals:
        return
    print("-" * 76)
    print(f"  games                        : {len(totals)}")
    print(f"  levels per game  mean/min/max: "
          f"{sum(counts)/len(counts):.1f} / {min(counts)} / {max(counts)}")
    print(f"  total levels                 : {sum(counts)}")
    print(f"  actions per LEVEL mean/min/max: "
          f"{sum(all_levels)/len(all_levels):.1f} / {min(all_levels)} / "
          f"{max(all_levels)}")
    print(f"  actions per GAME  mean       : {sum(totals)/len(totals):.1f}")
    print(f"  full-set human total         : {sum(totals)}")
    print()
    print("  cliff = 5x total. Past it a level scores ~4/100.")
    print("  Budget your agent against the per-level numbers, not the totals.")


def frame_cells(frame) -> list:
    out = []

    def walk(node, path):
        if isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, path + (i,))
        else:
            out.append((path, node))

    walk(frame, ())
    return out


def run_capture(arc, game_id: str, script: list) -> list:
    ScriptedAgent = build_scripted_agent_class()
    env = arc.make(game_id, render_mode=None)
    agent = ScriptedAgent(
        card_id="diag", game_id=game_id, agent_name=f"Diag.{game_id}",
        ROOT_URL="http://localhost", record=False, arc_env=env,
        tags=["diag"], script=script,
    )
    agent.main()
    return [getattr(fr, "frame", None) for fr in agent.frames]


def show_divergence(game_id: str, steps: int, seed: int) -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    if game_id not in envs:
        raise SystemExit(f"Unknown game {game_id!r}")

    script = build_script(steps, seed)
    print(f"Running {game_id} twice with an identical {len(script)}-step "
          f"script\n")
    a = run_capture(arc, game_id, script)
    b = run_capture(arc, game_id, script)
    n = min(len(a), len(b))

    print(f"{'step':>4} {'layers':>6} {'settled':>8} {'anim diff':>10}  note")
    print("-" * 52)

    settled_divergences, anim_only = [], []
    for i in range(n):
        la, lb = frame_layers(a[i]), frame_layers(b[i])
        same_settled = hash_frame(a[i]) == hash_frame(b[i])
        ca, cb = frame_cells(a[i]), frame_cells(b[i])
        if len(ca) == len(cb):
            anim = sum(1 for (_, x), (_, y) in zip(ca, cb) if x != y)
        else:
            anim = max(len(ca), len(cb))

        if not same_settled:
            settled_divergences.append(i)
        elif anim:
            anim_only.append(i)

        if anim or i < 3:
            note = ("SETTLED STATE DIFFERS" if not same_settled
                    else "animation only" if anim else "identical")
            print(f"{i:>4} {la:>3}/{lb:<3} "
                  f"{'same' if same_settled else 'DIFF':>8} {anim:>10}  {note}")

    print("\n========= INTERPRETATION =========")
    print(f"  steps compared                 : {n}")
    print(f"  steps where SETTLED state differs: {len(settled_divergences)}")
    print(f"  steps with animation-only diffs  : {len(anim_only)}")

    if not settled_divergences:
        print("\n  VERDICT: settled state never differed.")
        print("  The game is DETERMINISTIC. Earlier differences were purely")
        print("  intermediate animation frames, which do not affect logic.")
        print("  -> Safe to simulate and plan offline.")
        return

    first = settled_divergences[0]
    tail = [i for i in settled_divergences if i > first]
    if len(tail) > 0.5 * (n - first - 1):
        print(f"\n  VERDICT: settled state diverges from step {first} onward "
              f"and STAYS diverged.")
        print("  Genuine stochasticity. Plan with replanning + safety margin.")
    else:
        print(f"\n  VERDICT: settled state differed at step {first} but "
              f"recovered.")
        print("  Transient, not persistent randomness. Likely simulable.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--divergence", default=None, metavar="GAME")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    if args.baselines:
        show_baselines()
    elif args.divergence:
        show_divergence(args.divergence, args.steps, args.seed)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
