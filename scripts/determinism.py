"""Determinism experiment: does the same script produce the same frames?

This is the fork in the road for Phase 2.

  DETERMINISTIC  -> you can build an internal simulator, search it with BFS
                    or planning, and pay ZERO real actions for the search.
                    Actions become something you spend only on execution.

  STOCHASTIC     -> you need replanning with a safety margin against the
                    5x cliff, and any world model must carry uncertainty.

Method: run an identical fixed script N times from a fresh environment and
compare the sequence of frame hashes. Identical hash lists across all trials
means deterministic; the first divergent index tells you where and when
randomness enters (immediately = seeded per-episode, later = event-driven).

Usage:
    .venv/bin/python scripts/determinism.py --all --trials 3
    .venv/bin/python scripts/determinism.py --game ls20 --trials 5 --steps 150
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from lab import build_script, hash_frame, observe, write_jsonl
from probe import run_one

import arc_agi
from arc_agi import OperationMode


def compare(trials: list[list[str]]) -> dict:
    """Compare hash sequences across trials.

    Crucially this checks whether divergence PERSISTS. A game can differ at
    one step and then return to identical states -- that is animation or a
    transient, not stochastic game logic. Only divergence that continues is
    real. (Hashes here are settled-state hashes, so animation is already
    mostly filtered; this is the second line of defence.)
    """
    if len(trials) < 2:
        return {"verdict": "insufficient", "trials": len(trials)}

    lengths = {len(t) for t in trials}
    limit = min(len(t) for t in trials)

    diverged_at = [i for i in range(limit) if len({t[i] for t in trials}) > 1]
    first = diverged_at[0] if diverged_at else None

    # Does it persist? Look at everything after the first divergence.
    persisted = False
    if first is not None and first < limit - 1:
        tail = range(first + 1, limit)
        differing_tail = sum(1 for i in tail if len({t[i] for t in trials}) > 1)
        persisted = differing_tail > 0.5 * len(list(tail))

    if first is None and len(lengths) > 1:
        first = limit
        persisted = True

    if first is None:
        verdict = "deterministic"
    elif persisted:
        verdict = "stochastic"
    else:
        verdict = "transient"

    return {
        "verdict": verdict,
        "trials": len(trials),
        "lengths": sorted(lengths),
        "first_divergence": first,
        "diverging_steps": len(diverged_at),
        "persisted": persisted,
        "compared_steps": limit,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out", default="logs/determinism")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    outdir = Path(args.out)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = {e.game_id.split("-")[0]: e for e in arc.get_environments()}

    if args.all or not args.game:
        game_ids = sorted(envs)
    else:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        unknown = wanted - set(envs)
        if unknown:
            raise SystemExit(f"Unknown game id(s): {sorted(unknown)}")
        game_ids = sorted(wanted)

    script = build_script(args.steps, args.seed)
    print(f"Script: {len(script)} steps, seed={args.seed}")
    print(f"Testing {len(game_ids)} game(s) x {args.trials} trials\n")

    results = []
    for i, gid in enumerate(game_ids, 1):
        hash_seqs = []
        ok = True
        for _ in range(args.trials):
            try:
                steps, _, _ = run_one(arc, gid, script, envs.get(gid))
                hash_seqs.append([s.frame_hash for s in steps])
            except Exception as exc:
                print(f"[{i}/{len(game_ids)}] {gid:8} FAILED: {exc}")
                ok = False
                break
        if not ok:
            continue

        verdict = compare(hash_seqs)
        verdict["game_id"] = gid
        results.append(verdict)

        mark = {"deterministic": "OK ", "transient": "~~ ",
                "stochastic": "VAR"}.get(verdict["verdict"], "?? ")
        div = verdict["first_divergence"]
        print(f"[{i}/{len(game_ids)}] {gid:8} {mark} {verdict['verdict']:14}"
              + (f" first divergence at step {div}" if div is not None else ""))

    if not results:
        raise SystemExit("No games tested successfully.")

    write_jsonl(outdir / "determinism.jsonl", results)

    counts = Counter(r["verdict"] for r in results)
    det = counts.get("deterministic", 0) + counts.get("transient", 0)
    print("\n========= VERDICT =========")
    for k, v in counts.most_common():
        print(f"  {k:14} {v}/{len(results)} games")

    print()
    if counts.get("transient"):
        print("\n  'transient' = differed at one step then returned to identical.")
        print("  That is animation or a one-off, NOT stochastic game logic.")
        print("  Treat these as deterministic for planning purposes.")

    if det == len(results):
        print("\n  All games effectively deterministic.")
        print("  -> Phase 2: build an internal simulator. Search is FREE.")
        print("     Plan offline, then execute one near-optimal sequence.")
    elif det == 0:
        print("  No games deterministic.")
        print("  -> Phase 2: replanning architecture with uncertainty.")
        print("     Budget a safety margin against the 5x cliff.")
    else:
        det_games = [r["game_id"] for r in results
                     if r["verdict"] == "deterministic"]
        print(f"  MIXED. Deterministic: {', '.join(det_games)}")
        print("  -> Phase 2: simulator for the deterministic subset,")
        print("     replanning fallback for the rest. Detect at runtime.")

    divs = [r["first_divergence"] for r in results
            if r["first_divergence"] is not None]
    if divs:
        print(f"\n  Divergence steps: min={min(divs)} median="
              f"{sorted(divs)[len(divs)//2]} max={max(divs)}")
        if min(divs) <= 1:
            print("  Divergence at step 0-1 suggests randomised initial state.")
        else:
            print("  Late divergence suggests event-driven randomness;")
            print("  the early prefix may still be safely simulable.")

    print(f"\nRaw results -> {outdir}/determinism.jsonl")


if __name__ == "__main__":
    main()
