"""Read a probe run back, with the script's intent attached to each step.

The step log records WHICH action was taken but not the click coordinates or
why. Because scripted runs are deterministic and ordered, we can zip the
script onto the log and recover full context.

Usage:
    .venv/bin/python scripts/analyze_probe.py --game lf52 \
        --script probes/lf52_scan.json
    .venv/bin/python scripts/analyze_probe.py --game lf52 \
        --script probes/lf52_mech.json --verbose
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(game: str, script_path: str, logdir: str):
    script = json.loads(Path(script_path).read_text())
    steps_file = Path(logdir) / f"{game}.steps.jsonl"
    if not steps_file.exists():
        raise SystemExit(
            f"No log at {steps_file}. Run probe.py with this script first.")
    steps = [json.loads(l) for l in steps_file.open() if l.strip()]
    return script, steps


def pair(script: list[dict], steps: list[dict]):
    """Align script entries with the frames they produced.

    steps[0] is the initial frame before any scripted action, so
    script[i] corresponds to steps[i+1].
    """
    out = []
    for i, spec in enumerate(script):
        idx = i + 1
        if idx >= len(steps):
            break
        out.append((spec, steps[idx]))
    return out




def show_state_timeline(steps: list[dict]) -> None:
    """Report every state transition, and flag dead tails.

    A run that reaches GAME_OVER and never resets produces a long tail of
    no-ops. Without this you read that tail as "these actions do nothing",
    which is a statement about the harness, not the game.
    """
    transitions = []
    prev = None
    for st in steps:
        if st["state"] != prev:
            transitions.append((st["idx"], st["state"]))
            prev = st["state"]
    print("=== STATE TIMELINE ===")
    for idx, state in transitions:
        print(f"  step {idx:>4}  -> {state}")

    terminal = {"GAME_OVER", "WIN"}
    last_idx, last_state = transitions[-1]
    tail = len(steps) - 1 - last_idx
    if last_state in terminal and tail > 5:
        print(f"\n  WARNING: run ended in {last_state} at step {last_idx} "
              f"with {tail} steps after it.")
        print("  Those trailing steps are no-ops against a dead environment.")
        print("  Any 'unresponsive' region past that point is an artifact.")
    print()


def render_map(script: list[dict], pairs) -> None:
    """ASCII map of which scanned cells responded.

    A coordinate list tells you the bounding box; a map tells you the SHAPE.
    Those are very different pieces of information -- a hollow ring and a
    solid block have identical bounding boxes.
    """
    cells = {}
    dead_from = None
    for spec, st in pairs:
        if st["state"] in ("GAME_OVER", "WIN") and dead_from is None:
            dead_from = st["idx"]
        if spec.get("phase") != "scan" or "x" not in spec:
            continue
        if dead_from is not None and st["idx"] > dead_from:
            cells[(spec["x"], spec["y"])] = "dead"
        else:
            cells[(spec["x"], spec["y"])] = bool(st["changed"])
    if not cells:
        return

    xs = sorted({c[0] for c in cells})
    ys = sorted({c[1] for c in cells})
    print("\n=== RESPONSE MAP  (# = changed, . = no-op, x = after "
          "terminal state) ===")
    print("      " + "".join(f"{x%10}" for x in xs))
    for y in ys:
        row = "".join(
            "x" if cells.get((x, y)) == "dead"
            else "#" if cells.get((x, y)) else "." for x in xs)
        print(f"  y={y:<3} {row}")
    print(f"      x = {xs[0]}..{xs[-1]} step {xs[1]-xs[0] if len(xs)>1 else 1}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--logdir", default="logs/probe")
    p.add_argument("--verbose", action="store_true",
                   help="Print every step, not just responsive ones.")
    args = p.parse_args()

    script, steps = load(args.game, args.script, args.logdir)
    pairs = pair(script, steps)
    if not pairs:
        raise SystemExit("Script and log did not align. Re-run the probe.")

    print(f"{len(pairs)} scripted actions matched to frames\n")

    by_phase = collections.defaultdict(lambda: {"n": 0, "changed": 0,
                                                "hits": []})
    level_events = []
    prev_levels = steps[0].get("levels_completed", 0)

    for spec, st in pairs:
        ph = spec.get("phase", "?")
        rec = by_phase[ph]
        rec["n"] += 1
        if st["changed"]:
            rec["changed"] += 1
            rec["hits"].append((spec, st))
        if st["levels_completed"] > prev_levels:
            level_events.append((spec, st))
            prev_levels = st["levels_completed"]

    print(f"{'phase':16} {'actions':>8} {'changed':>8} {'rate':>7}")
    print("-" * 44)
    for ph, rec in by_phase.items():
        rate = rec["changed"] / rec["n"] if rec["n"] else 0
        print(f"{ph:16} {rec['n']:>8} {rec['changed']:>8} {rate:>6.0%}")

    show_state_timeline(steps)

    scan = by_phase.get("scan")
    if scan:
        render_map(script, pairs)
    if scan and scan["hits"]:
        cells = [(s["x"], s["y"]) for s, _ in scan["hits"]
                 if "x" in s and "y" in s]
        print(f"\n=== RESPONSIVE CELLS ({len(cells)} of {scan['n']}) ===")
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        if cells:
            print(f"  x range {min(xs)}-{max(xs)}, y range {min(ys)}-{max(ys)}")
            print(f"  first 20: {cells[:20]}")
            print("\n  Use two of these for --cell and --cell2 in "
                  "make_probe.py --mechanics")
    elif scan:
        print("\n  No cell responded. Try a smaller --stride, or the game "
              "may need a different action first.")

    if args.verbose or by_phase.keys() - {"scan", "init"}:
        print("\n=== STEP DETAIL (state-changing only) ===")
        for spec, st in pairs:
            if spec.get("phase") == "scan" and not args.verbose:
                continue
            if not st["changed"] and not args.verbose:
                continue
            coord = (f" ({spec['x']},{spec['y']})"
                     if "x" in spec else "")
            flag = "CHANGED" if st["changed"] else "no-op  "
            print(f"  [{spec.get('phase','?'):14}] {spec['action']:8}{coord:10} "
                  f"{flag}  lvl={st['levels_completed']} "
                  f"state={st['state']} layers={st.get('layers','?')}")

    if level_events:
        print(f"\n=== LEVEL COMPLETIONS ({len(level_events)}) ===")
        for spec, st in level_events:
            print(f"  {spec['action']} in phase {spec.get('phase')} "
                  f"-> level {st['levels_completed']}")

    print("\n=== READING THIS ===")
    print("  A phase at 0% changed  : those controls do nothing in this")
    print("                           context (wrong mode, or not yet legal).")
    print("  A phase at 100% changed: every action registers -- but check")
    print("                           whether it ACCOMPLISHES anything or is")
    print("                           just selection feedback.")
    print("  C_toggle alternating   : clicking toggles selection on/off.")
    print("  D_click_arrow active   : arrows steer a selected piece.")
    print("  E_two_click active     : moves are source-then-destination.")


if __name__ == "__main__":
    main()
