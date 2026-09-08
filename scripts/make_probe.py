"""Generate hand-authored probe scripts.

Random scripts tell you almost nothing. A probe script is an EXPERIMENT:
every action is there to answer one question, and questions are grouped into
phases so the log can be read back meaningfully.

Two modes:

  --scan     Coarse click sweep across the grid to find which cells respond
             at all. Run this FIRST -- you cannot write a targeted probe
             until you know where the live cells are.

  --mechanics  Phase-structured test of control semantics: do arrows move a
             cursor or a piece, does clicking toggle, what does ACTION7 do,
             is a "move" one action or several.

Usage:
    .venv/bin/python scripts/make_probe.py --scan --stride 4 \
        --out probes/lf52_scan.json
    .venv/bin/python scripts/make_probe.py --mechanics --cell 20,20 \
        --cell2 24,24 --out probes/lf52_mech.json

Then:
    .venv/bin/python scripts/probe.py --game lf52 --script probes/lf52_scan.json
    .venv/bin/python scripts/analyze_probe.py --game lf52 \
        --script probes/lf52_scan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan_script(stride: int, grid: int, actions: list[str]) -> list[dict]:
    """Click a coarse lattice across the grid.

    Locally this is free (~2700 actions/sec), so prefer a dense-ish stride.
    The point is to find WHICH cells do anything, not to play well.
    """
    script: list[dict] = [{"action": "RESET", "phase": "init", "why": "start"}]
    if "ACTION6" not in actions:
        raise SystemExit("Scan mode needs ACTION6 (click) in the action space.")
    for y in range(0, grid, stride):
        for x in range(0, grid, stride):
            script.append({
                "action": "ACTION6", "x": x, "y": y,
                "phase": "scan", "why": f"probe cell ({x},{y})",
            })
    return script


def mechanics_script(cell_a: tuple[int, int], cell_b: tuple[int, int],
                     actions: list[str]) -> list[dict]:
    """Phase-structured control-semantics probe.

    Each phase isolates ONE question. Phases are separated so the analyzer
    can report them independently. Total cost is small (~30 actions), which
    matters if you ever run this against a scored environment.
    """
    ax, ay = cell_a
    bx, by = cell_b
    has = set(actions)
    s: list[dict] = [{"action": "RESET", "phase": "init", "why": "start"}]

    def add(action: str, phase: str, why: str, **kw) -> None:
        if action in has:
            s.append({"action": action, "phase": phase, "why": why, **kw})

    # A: do arrow keys do anything on their own, and do they accumulate?
    # If a cursor exists, repeated same-direction presses should keep moving
    # until a boundary. If they move a piece, expect different behaviour.
    for i in (1, 2):
        add("ACTION1", "A_arrows", f"up x{i}")
    for i in (1, 2):
        add("ACTION2", "A_arrows", f"down x{i}")
    add("ACTION3", "A_arrows", "left")
    add("ACTION4", "A_arrows", "right")

    # B: what is ACTION7? Twice in a row reveals whether it is idempotent
    # (a mode toggle) or a no-op without prior selection.
    add("ACTION7", "B_action7", "bare press 1")
    add("ACTION7", "B_action7", "bare press 2")

    # C: does clicking the same cell twice toggle selection on and off?
    add("ACTION6", "C_toggle", "click A", x=ax, y=ay)
    add("ACTION6", "C_toggle", "click A again", x=ax, y=ay)
    add("ACTION6", "C_toggle", "click A third", x=ax, y=ay)

    # D: after a click, do arrows behave differently? This is the key test
    # for "select then steer" versus "select then click destination".
    add("ACTION6", "D_click_arrow", "click A", x=ax, y=ay)
    add("ACTION1", "D_click_arrow", "arrow after click")
    add("ACTION4", "D_click_arrow", "arrow after click 2")
    add("ACTION7", "D_click_arrow", "confirm after click+arrow")

    # E: is a move click-source-then-click-destination? Then try to reverse
    # it. If B->A undoes A->B, moves are free-form; if not, there are rules.
    add("ACTION6", "E_two_click", "click source A", x=ax, y=ay)
    add("ACTION6", "E_two_click", "click dest B", x=bx, y=by)
    add("ACTION6", "E_two_click", "click source B", x=bx, y=by)
    add("ACTION6", "E_two_click", "click dest A", x=ax, y=ay)

    # F: does clicking empty space cancel a pending selection?
    add("ACTION6", "F_cancel", "click A", x=ax, y=ay)
    add("ACTION6", "F_cancel", "click far corner", x=1, y=1)
    add("ACTION7", "F_cancel", "confirm after cancel")

    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--mechanics", action="store_true")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--cell", default="20,20", help="Primary cell, 'x,y'.")
    p.add_argument("--cell2", default="24,24", help="Secondary cell, 'x,y'.")
    p.add_argument("--actions", default="ACTION1,ACTION2,ACTION3,ACTION4,"
                                        "ACTION6,ACTION7",
                   help="Legal actions for this game (from probe output).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    if args.scan:
        script = scan_script(args.stride, args.grid, actions)
    elif args.mechanics:
        ca = tuple(int(v) for v in args.cell.split(","))
        cb = tuple(int(v) for v in args.cell2.split(","))
        script = mechanics_script(ca, cb, actions)
    else:
        raise SystemExit("Pass --scan or --mechanics.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(script, indent=1))
    phases = {}
    for step in script:
        phases[step.get("phase", "?")] = phases.get(step.get("phase", "?"), 0) + 1
    print(f"Wrote {len(script)} steps -> {out}")
    for name, count in phases.items():
        print(f"  {name:16} {count:>4} actions")


if __name__ == "__main__":
    main()
