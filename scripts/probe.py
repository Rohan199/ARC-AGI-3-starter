"""Probe harness: run a FIXED action script against games and log everything.

This is not an agent. It is the measuring instrument every Phase 1
experiment runs through. Nothing here decides what to do -- you hand it a
script, it executes it and records what happened.

Usage:
    .venv/bin/python scripts/probe.py --game ls20 --steps 120
    .venv/bin/python scripts/probe.py --all --steps 80 --out logs/baseline
    .venv/bin/python scripts/probe.py --game ls20 --script probes/wall_test.json

Outputs (per game) into --out:
    <game>.steps.jsonl   one row per frame
    <game>.summary.json  waste ratio, action space, levels, RHAE score
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lab
from lab import (
    build_scripted_agent_class,
    build_script,
    observe,
    observed_action_space,
    score_run,
    segment_levels,
    waste_ratio,
    write_jsonl,
)

import arc_agi
from arc_agi import OperationMode


def run_one(arc, game_id: str, script: list[dict], env_info=None, render=None):
    """Play one scripted run. Returns (steps, levels, score_result)."""
    ScriptedAgent = build_scripted_agent_class()
    env = arc.make(game_id, render_mode=render)
    if env is None:
        raise RuntimeError(f"could not create env for {game_id!r}")

    agent = ScriptedAgent(
        card_id="probe",
        game_id=game_id,
        agent_name=f"Probe.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["probe"],
        script=script,
    )
    agent.main()

    steps = observe(agent.frames)
    final_state = steps[-1].state if steps else "UNKNOWN"
    levels = segment_levels(steps, final_state)
    score = score_run(game_id, levels, env_info=env_info, final_state=final_state)
    return steps, levels, score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="Game id, or comma-separated list.")
    p.add_argument("--all", action="store_true", help="Probe every available game.")
    p.add_argument("--steps", type=int, default=80, help="Script length.")
    p.add_argument("--seed", type=int, default=1234, help="Script seed (fixed!).")
    p.add_argument("--script", default=None,
                   help="Path to a JSON list of action dicts, e.g. "
                        '[{"action":"ACTION1"},{"action":"ACTION6","x":3,"y":9}]')
    p.add_argument("--out", default="logs/probe", help="Output directory.")
    p.add_argument("--render", default=None, choices=[None, "terminal"])
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

    if args.script:
        script = json.loads(Path(args.script).read_text())
        if not script or script[0].get("action") != "RESET":
            script = [{"action": "RESET", "why": "auto-prepended"}] + script
        print(f"Script: {len(script)} steps from {args.script}")
    else:
        script = build_script(args.steps, args.seed)
        print(f"Script: {len(script)} steps, seed={args.seed} (reproducible)")

    print(f"Probing {len(game_ids)} game(s) -> {outdir}\n")
    rows = []
    any_estimated = False

    for i, gid in enumerate(game_ids, 1):
        try:
            steps, levels, score = run_one(arc, gid, script, envs.get(gid), args.render)
        except Exception as exc:
            print(f"[{i}/{len(game_ids)}] {gid:8} FAILED: {exc}")
            continue

        write_jsonl(outdir / f"{gid}.steps.jsonl",
                    [s.__dict__ for s in steps])
        summary = {
            "game_id": gid,
            "steps": len(steps),
            "waste_ratio": round(waste_ratio(steps), 4),
            "action_space": observed_action_space(steps),
            "levels_completed": score.levels_completed,
            "rhae": round(score.score, 3),
            "rhae_estimated": score.estimated,
            "final_state": steps[-1].state if steps else "UNKNOWN",
            "level_actions": score.level_actions,
            "baselines": score.baselines,
        }
        (outdir / f"{gid}.summary.json").write_text(json.dumps(summary, indent=2))
        rows.append(summary)
        any_estimated |= score.estimated

        print(f"[{i}/{len(game_ids)}] {gid:8} "
              f"waste={summary['waste_ratio']:.1%}  "
              f"levels={summary['levels_completed']}  "
              f"rhae={summary['rhae']:.2f}"
              f"{'*' if score.estimated else ''}  "
              f"actions={observed_action_space(steps)}")

    if not rows:
        raise SystemExit("No games probed successfully.")

    print("\n========= AGGREGATE =========")
    mean_waste = sum(r["waste_ratio"] for r in rows) / len(rows)
    mean_rhae = sum(r["rhae"] for r in rows) / len(rows)
    spaces = {tuple(r["action_space"]) for r in rows}
    print(f"  games probed      : {len(rows)}")
    print(f"  mean waste ratio  : {mean_waste:.1%}")
    print(f"  mean RHAE         : {mean_rhae:.3f}")
    print(f"  distinct action spaces across games: {len(spaces)}")
    for sp in sorted(spaces):
        who = [r['game_id'] for r in rows if tuple(r['action_space']) == sp]
        print(f"    {list(sp)}  <- {len(who)} game(s): {', '.join(who[:6])}"
              f"{'...' if len(who) > 6 else ''}")
    if any_estimated:
        print("\n  * RHAE marked with * uses ESTIMATED baselines "
              f"({lab.AVO_FALLBACK_BASELINE} actions/level, AVO-derived).")
        print("    Real baselines were not in the environment metadata for "
              "some games.")
    print(f"\nPer-step logs written to {outdir}/")


if __name__ == "__main__":
    main()
