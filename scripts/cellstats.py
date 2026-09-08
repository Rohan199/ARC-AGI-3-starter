"""Per-cell change statistics: separate real state from counters and UI.

Why this exists
---------------
Whole-frame change detection fails on any game that renders a move counter,
timer or turn indicator into the grid, because those cells change on EVERY
action. The frame always differs, so "did anything change?" always answers
yes and tells you nothing.

The fix is to measure change per CELL rather than per frame:

  ~100% change rate  -> counter, clock, animation. Noise. Mask it.
  0% change rate     -> static background or border. Ignore it.
  in between         -> actual game state. This is the signal.

Once you know the mask, "meaningful change" becomes computable, and so does
a sane click-candidate set.

Usage:
    .venv/bin/python scripts/cellstats.py --game lf52
    .venv/bin/python scripts/cellstats.py --game lf52 \
        --script probes/lf52_scan.json
    .venv/bin/python scripts/cellstats.py --all --steps 120
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lab import _settled, build_script, build_scripted_agent_class

import arc_agi
from arc_agi import OperationMode


def collect_grids(arc, game_id: str, script: list[dict]) -> tuple[list, list]:
    ScriptedAgent = build_scripted_agent_class()
    env = arc.make(game_id, render_mode=None)
    agent = ScriptedAgent(
        card_id="cells", game_id=game_id, agent_name=f"Cells.{game_id}",
        ROOT_URL="http://localhost", record=False, arc_env=env,
        tags=["cells"], script=script,
    )
    agent.main()
    grids = [_settled(getattr(f, "frame", None)) for f in agent.frames]
    states = [getattr(f.state, "name", str(f.state)) for f in agent.frames]
    return grids, states


def episode_lengths(states: list[str]) -> list[int]:
    """Actions between entering play and hitting a terminal state.

    A consistent number here means the GAME enforces an action limit, which
    is a much harder bound than RHAE's 5x cliff and must be designed around.
    """
    lengths, start = [], None
    for i, st in enumerate(states):
        if st == "NOT_FINISHED" and start is None:
            start = i
        elif st in ("GAME_OVER", "WIN") and start is not None:
            lengths.append(i - start)
            start = None
    return lengths


def cell_change_rates(grids: list) -> dict[tuple[int, int], float]:
    """Fraction of consecutive frame pairs in which each cell changed."""
    usable = [g for g in grids if isinstance(g, list) and g]
    if len(usable) < 2:
        return {}
    counts: Counter = Counter()
    pairs = 0
    for a, b in zip(usable, usable[1:]):
        if len(a) != len(b):
            continue
        pairs += 1
        for r, (ra, rb) in enumerate(zip(a, b)):
            if ra == rb:
                continue
            for c, (va, vb) in enumerate(zip(ra, rb)):
                if va != vb:
                    counts[(r, c)] += 1
    if not pairs:
        return {}
    return {cell: n / pairs for cell, n in counts.items()}


def render(rates: dict, rows: int, cols: int, stride: int = 2) -> None:
    """ASCII volatility map. Symbol shows how often each cell changes."""
    def sym(v: float) -> str:
        if v == 0:
            return "."
        if v >= 0.95:
            return "@"     # changes almost every action -> counter/clock
        if v >= 0.5:
            return "%"
        if v >= 0.15:
            return "+"
        return "-"         # rare change -> likely real game state

    print("\n=== CELL VOLATILITY  (. never  - rare  + some  % often  "
          "@ every action) ===")
    for r in range(0, rows, stride):
        line = "".join(sym(rates.get((r, c), 0.0)) for c in range(0, cols, stride))
        print(f"  r={r:<3} {line}")
    print(f"      c = 0..{cols - 1} step {stride}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--script", default=None)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default="logs/cellstats")
    args = p.parse_args()

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = {e.game_id.split("-")[0]: e for e in arc.get_environments()}
    games = sorted(envs) if (args.all or not args.game) else \
        [g.strip() for g in args.game.split(",")]

    script = (json.loads(Path(args.script).read_text()) if args.script
              else build_script(args.steps, args.seed))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for gid in games:
        if gid not in envs:
            print(f"skip unknown game {gid}")
            continue
        grids, states = collect_grids(arc, gid, script)
        eps = episode_lengths(states)
        rates = cell_change_rates(grids)
        if not rates:
            print(f"{gid}: no usable grid pairs")
            continue

        rows = len(grids[1]) if len(grids) > 1 and grids[1] else 0
        cols = len(grids[1][0]) if rows else 0
        always = [c for c, v in rates.items() if v >= 0.95]
        real = [c for c, v in rates.items() if 0 < v < 0.95]

        print(f"\n{'=' * 60}\n{gid}   grid {rows}x{cols}")
        if eps:
            uniq = sorted(set(eps))
            print(f"  episode lengths: {eps}")
            if len(uniq) == 1:
                print(f"  >> HARD ACTION LIMIT: {uniq[0]} actions per "
                      f"episode.")
                b = list(getattr(envs[gid], "baseline_actions", None) or [])
                if b:
                    print(f"  >> level-1 baseline {b[0]} -> limit is "
                          f"{uniq[0]/b[0]:.1f}x baseline")
                    print(f"  >> best possible if you use the whole limit: "
                          f"{min(115, (b[0]/uniq[0])**2*100):.0f}/100")
        else:
            print("  no terminal state reached in this run")

        print(f"  cells that ever changed : {len(rates)}")
        print(f"  changing ~every action  : {len(always)}  <- mask these")
        print(f"  changing sometimes      : {len(real)}  <- real state")
        if always:
            rs = sorted({c[0] for c in always})
            cs = sorted({c[1] for c in always})
            print(f"  noise rows {rs[0]}-{rs[-1]}, cols {cs[0]}-{cs[-1]}")

        if len(games) == 1:
            render(rates, rows, cols, args.stride)

        (outdir / f"{gid}.cells.json").write_text(json.dumps({
            "game_id": gid, "rows": rows, "cols": cols,
            "episode_lengths": eps,
            "mask_cells": [list(c) for c in sorted(always)],
            "signal_cells": len(real),
        }, indent=1))
        summary.append((gid, eps[0] if eps else None, len(always), len(real)))

    if len(summary) > 1:
        print(f"\n{'=' * 60}\n{'game':6} {'ep limit':>9} {'noise':>7} "
              f"{'signal':>8}")
        for gid, ep, noise, real in summary:
            print(f"{gid:6} {str(ep):>9} {noise:>7} {real:>8}")
        print("\nA consistent 'ep limit' means the GAME caps your actions.")
        print("That bound is usually tighter than RHAE's 5x cliff.")

    print(f"\nMasks written to {outdir}/")


if __name__ == "__main__":
    main()
