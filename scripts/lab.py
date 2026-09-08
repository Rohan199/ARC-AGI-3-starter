"""Phase 1 instrumentation library for ARC-AGI-3.

Everything here is measurement, not strategy. Three jobs:

  1. Drive a *fixed, scripted* action sequence at a game (ScriptedAgent).
  2. Turn a run into structured per-step records (observe).
  3. Score a run with the OFFICIAL RHAE calculator (score_run).

Design notes
------------
* We never reimplement RHAE. `arc_agi.EnvironmentScoreCalculator` is the
  same code the leaderboard uses; a local reimplementation would silently
  drift. We only feed it correctly-segmented level data.

* Per-level human baselines ship in `EnvironmentInfo.baseline_actions`.
  When present we use them verbatim. When absent we fall back to an
  estimate and mark the result `estimated=True` so you never mistake a
  guess for a real score.

* Frames are hashed rather than stored. A 64x64 grid per step across
  hundreds of steps and 25 games is a lot of JSON; a hash answers
  "did this change?" and "is this run identical to that one?" which is
  all Phase 1 needs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
for _p in (ROOT, VENDOR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

# AVO (NVIDIA, Aug 2026) cleared 183 levels in 6,624 actions on the 25-game
# public set at ~100 RHAE. 6624/183 ~= 36.2 actions/level. We use that as a
# stand-in ONLY when real baselines are unavailable. It is an average across
# very different games, so treat any score derived from it as a rough signal.
AVO_FALLBACK_BASELINE = 36

# At 5x baseline the score is (1/5)^2 * 100 = 4.0. Useful as a "danger line"
# when budgeting actions, even though the calculator itself has no hard cutoff.
CLIFF_MULTIPLIER = 5


# --------------------------------------------------------------------------
# Frame hashing
# --------------------------------------------------------------------------

def _settled(frame):
    """Return the settled (final) grid from a frame response.

    A FrameData.frame is a LIST OF GRIDS, not one grid. Most steps return 2,
    but an action that triggers an animation can return 20+ intermediate
    grids. Only the last one is the state the game actually settled into --
    the rest are cosmetic in-between frames whose exact contents can vary
    between otherwise identical runs.

    Hashing all of them (which this library did originally) makes animated
    games look stochastic and makes every animated step look "changed".
    """
    if not isinstance(frame, list) or not frame:
        return frame
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return frame[-1]
    return frame


def frame_layers(frame) -> int:
    """How many grids came back in this frame response."""
    if not isinstance(frame, list) or not frame:
        return 0
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return len(frame)
    return 1


def hash_frame(frame: Any, settled: bool = True) -> str:
    """Stable short hash of a frame.

    settled=True (default) hashes ONLY the final grid -- the real state.
    settled=False hashes everything including animation, which is almost
    never what you want but is useful for diagnosing animation itself.
    """
    if frame is None:
        return "none"
    target = _settled(frame) if settled else frame
    payload = json.dumps(target, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# Scripted agent
# --------------------------------------------------------------------------

def make_action(spec: dict[str, Any]) -> GameAction:
    """Materialise a GameAction from a plain dict.

    We keep scripts as dicts rather than GameAction objects because
    GameAction members are shared enum singletons -- calling set_data()
    mutates the member itself, so reusing one across runs would leak
    coordinates between trials and quietly break determinism testing.
    """
    action = GameAction[spec["action"]]
    if action.is_complex():
        action.set_data({"x": int(spec.get("x", 0)), "y": int(spec.get("y", 0))})
    action.reasoning = spec.get("why", "scripted probe")
    return action


def build_script(
    n: int,
    seed: int,
    actions: Optional[Iterable[str]] = None,
    grid: int = 64,
) -> list[dict[str, Any]]:
    """Deterministic pseudo-random script. Same seed -> same script, always.

    Note this is *our* randomness, fixed by seed. It is not the agent being
    random at runtime -- that distinction is the whole point of the
    determinism experiment.
    """
    import random as _random

    rng = _random.Random(seed)
    pool = list(actions) if actions else [
        a.name for a in GameAction if a is not GameAction.RESET
    ]
    script: list[dict[str, Any]] = [{"action": "RESET", "why": "script start"}]
    for _ in range(n):
        name = rng.choice(pool)
        step: dict[str, Any] = {"action": name}
        if GameAction[name].is_complex():
            step["x"] = rng.randint(0, grid - 1)
            step["y"] = rng.randint(0, grid - 1)
        script.append(step)
    return script


def _load_agent_base():
    try:
        from agents.agent import Agent  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Could not import agents.agent.Agent. Run `make setup` so the "
            f"framework lands in {VENDOR}."
        ) from exc
    return Agent


def build_scripted_agent_class():
    """Build ScriptedAgent lazily so importing lab.py never requires vendor/."""
    Agent = _load_agent_base()

    class ScriptedAgent(Agent):
        """Plays a fixed list of actions. No randomness, no strategy."""

        MAX_ACTIONS = 100_000  # the script length is the real bound

        def __init__(self, *args: Any, script: Optional[list] = None,
                     auto_reset: bool = True, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.script: list[dict[str, Any]] = list(script or [])
            self.ptr = 0
            self.exhausted = False
            # Without this, hitting GAME_OVER mid-script leaves the agent
            # firing no-ops into a dead environment for the rest of the run,
            # which looks exactly like "those actions did nothing".
            self.auto_reset = auto_reset
            self.resets: list[int] = []

        def is_done(self, frames, latest_frame) -> bool:
            if latest_frame.state is GameState.WIN:
                return True
            return self.ptr >= len(self.script)

        def choose_action(self, frames, latest_frame) -> GameAction:
            # The engine requires a RESET out of NOT_PLAYED. We honour that
            # without consuming a script slot, so trials stay aligned.
            if latest_frame.state is GameState.NOT_PLAYED:
                if self.ptr < len(self.script) and \
                        self.script[self.ptr]["action"] == "RESET":
                    self.ptr += 1
                return make_action({"action": "RESET", "why": "engine requires reset"})

            if latest_frame.state is GameState.GAME_OVER and self.auto_reset:
                self.resets.append(self.ptr)
                return make_action({"action": "RESET", "why": "recover from game over"})

            if self.ptr >= len(self.script):
                self.exhausted = True
                return make_action({"action": "RESET", "why": "script exhausted"})

            spec = self.script[self.ptr]
            self.ptr += 1
            return make_action(spec)

    return ScriptedAgent


# --------------------------------------------------------------------------
# Observation records
# --------------------------------------------------------------------------

@dataclass
class Step:
    idx: int
    action: Optional[str]
    frame_hash: str
    changed: bool
    state: str
    levels_completed: int
    available_actions: list[int] = field(default_factory=list)
    full_reset: bool = False
    layers: int = 0


def observe(frames: list[FrameData]) -> list[Step]:
    """Turn a finished run's frame list into per-step records.

    We derive everything post-hoc from `agent.frames` rather than logging
    inside choose_action, because FrameData already carries the action that
    produced it (`action_input`) plus state, level count and legal actions.
    Fewer moving parts, and it cannot desync from what the engine saw.
    """
    steps: list[Step] = []
    prev_hash: Optional[str] = None
    for i, fr in enumerate(frames):
        raw = getattr(fr, "frame", None)
        h = hash_frame(raw)
        action = None
        ai = getattr(fr, "action_input", None)
        if ai is not None:
            action = getattr(ai, "name", None) or getattr(ai, "id", None) or str(ai)
        steps.append(
            Step(
                idx=i,
                action=str(action) if action is not None else None,
                frame_hash=h,
                changed=(prev_hash is not None and h != prev_hash),
                state=getattr(fr.state, "name", str(fr.state)),
                levels_completed=int(getattr(fr, "levels_completed", 0) or 0),
                available_actions=list(getattr(fr, "available_actions", []) or []),
                full_reset=bool(getattr(fr, "full_reset", False)),
                layers=frame_layers(raw),
            )
        )
        prev_hash = h
    return steps


def waste_ratio(steps: list[Step]) -> float:
    """Fraction of actions that produced no observable frame change.

    This is the single most useful Phase 1 number. For a random agent it is
    typically very high; driving it down is most of the RHAE battle.
    """
    acted = [s for s in steps if s.idx > 0]
    if not acted:
        return 0.0
    return sum(1 for s in acted if not s.changed) / len(acted)


def observed_action_space(steps: list[Step]) -> list[int]:
    """Union of `available_actions` seen across a run.

    Answers "is the action space the same for every game?" empirically
    instead of assuming ACTION1-7 everywhere.
    """
    seen: set[int] = set()
    for s in steps:
        seen.update(s.available_actions)
    return sorted(seen)


# --------------------------------------------------------------------------
# Level segmentation
# --------------------------------------------------------------------------

@dataclass
class LevelRecord:
    level_index: int      # 1-indexed; the calculator uses this as the weight
    completed: bool
    actions_taken: int


def segment_levels(steps: list[Step], final_state: str) -> list[LevelRecord]:
    """Split a run into per-level action counts.

    A level ends when `levels_completed` increments. Actions spent on failed
    attempts before a successful one still belong to that level -- that is
    what the scorer charges you for, so we count them.

    The trailing partial level (attempted but never cleared) is emitted with
    completed=False so it scores 0 but still contributes its weight, which is
    how the official calculator applies the completion cap.
    """
    records: list[LevelRecord] = []
    level_idx = 1
    actions_in_level = 0
    prev_levels = 0

    for s in steps:
        if s.idx == 0:
            prev_levels = s.levels_completed
            continue
        actions_in_level += 1
        if s.levels_completed > prev_levels:
            for _ in range(s.levels_completed - prev_levels):
                records.append(LevelRecord(level_idx, True, actions_in_level))
                level_idx += 1
                actions_in_level = 0
            prev_levels = s.levels_completed

    if actions_in_level > 0 and final_state != "WIN":
        records.append(LevelRecord(level_idx, False, actions_in_level))
    return records


# --------------------------------------------------------------------------
# Scoring (official calculator)
# --------------------------------------------------------------------------

@dataclass
class ScoreResult:
    game_id: str
    score: float
    levels_completed: int
    actions: int
    estimated: bool
    baselines: list[int]
    level_scores: list[float]
    level_actions: list[int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_baselines(
    env_info: Any, n_levels: int, fallback: int = AVO_FALLBACK_BASELINE
) -> tuple[list[int], bool]:
    """Get per-level human baselines, or fall back with a loud flag."""
    raw = list(getattr(env_info, "baseline_actions", None) or [])
    raw = [b for b in raw if isinstance(b, int) and b > 0]
    if len(raw) >= n_levels:
        return raw[:n_levels], False
    if raw:
        # Partial data: use what we have, extend with the mean of the rest.
        mean = max(1, round(sum(raw) / len(raw)))
        return raw + [mean] * (n_levels - len(raw)), True
    return [fallback] * n_levels, True


def score_run(
    game_id: str,
    levels: list[LevelRecord],
    env_info: Any = None,
    final_state: Optional[str] = None,
) -> ScoreResult:
    """Score one game's run using the official EnvironmentScoreCalculator."""
    from arc_agi import EnvironmentScoreCalculator

    baselines, estimated = resolve_baselines(env_info, len(levels))
    calc = EnvironmentScoreCalculator(id=game_id, state=final_state)
    for rec, base in zip(levels, baselines):
        calc.add_level(
            level_index=rec.level_index,
            completed=rec.completed,
            actions_taken=rec.actions_taken,
            baseline_actions=base,
            game_id=game_id,
        )
    es = calc.to_score(include_levels=True)
    return ScoreResult(
        game_id=game_id,
        score=float(es.score),
        levels_completed=int(es.levels_completed),
        actions=int(es.actions),
        estimated=estimated,
        baselines=baselines,
        level_scores=list(es.level_scores or []),
        level_actions=list(es.level_actions or []),
    )


# --------------------------------------------------------------------------
# Decision support
# --------------------------------------------------------------------------

def actions_for_target(baseline: int, target_score: float) -> int:
    """How many actions may I spend and still hit `target_score` on a level?

    Inverts score = (baseline/actions)^2 * 100.
    """
    if target_score <= 0:
        return 10**9
    return max(1, int(baseline / (min(target_score, 115.0) / 100.0) ** 0.5))


def cliff_actions(baseline: int) -> int:
    """Action count at which a level is worth ~4 points. Budget below this."""
    return baseline * CLIFF_MULTIPLIER


def marginal_value(baseline: int, actions: int) -> float:
    """Points gained by removing one action at the current count.

    Because the metric is squared, savings are worth far more when you are
    already efficient. This tells you where optimisation effort pays.
    """
    if actions <= 1:
        return 0.0
    now = min(115.0, (baseline / actions) ** 2 * 100)
    better = min(115.0, (baseline / (actions - 1)) ** 2 * 100)
    return better - now


# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
