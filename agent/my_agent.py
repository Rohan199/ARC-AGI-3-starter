"""ARC-AGI-3 agent: novelty-seeking search over a state graph.

This is the ONLY file that ships to Kaggle. `scripts/build_notebook.py`
copies its text into the submission notebook, so everything must be
self-contained -- no importing from scripts/lab.py, which does not exist
in the sandbox.

Design, and why
---------------
Measured facts from Phase 1 that drive this:

* Legal actions differ per game AND change within a run. There are 10
  distinct action sets across the 25 local games, and some games expose
  only ACTION6. So we read `available_actions` every frame instead of
  assuming ACTION1-7 (the template's uniform random over all actions
  wastes a large fraction of its budget on illegal moves).

* ACTION6 carries x/y, so its true branching factor is ~4096. Random
  clicking cannot work -- the random baseline wasted 98.8% of actions on
  lp85. We derive a small candidate set from objects in the grid instead.

* A frame is a LIST of grids (animation). Only the last is the settled
  state; hashing all of them makes animated games look random.

* Games are deterministic. One observation of (state, action) -> state'
  is ground truth, so the graph never needs repeated sampling.

* Several games enforce their own action limit (1.8x to 4.5x the human
  baseline) and end in GAME_OVER. We must reset and retry, and we learn
  the limit from experience.

* Scoring is RHAE: (baseline/actions)^2. Completing a level slowly is
  worth little, but completing nothing is worth zero. This agent aims at
  COMPLETION first; efficiency is the next iteration's problem.

Strategy: maintain a graph of settled states. From the current state,
prefer actions never tried here. Break ties toward actions with a good
historical record of changing something. Avoid transitions already known
to be self-loops. Reset on death, and give up if deaths stop producing
progress so wall-clock is left for other games.
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import Counter, defaultdict
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


# GameAction members are declared as tuples -- ACTION6 = (6, ComplexAction) --
# and __init__ reassigns _value_ afterwards. The enum's own lookup map still
# holds the tuple keys, so GameAction(6) raises ValueError even though
# GameAction.ACTION6.value == 6. Build an explicit id -> member map instead.
_BY_ID = {a.value: a for a in GameAction}


def action_by_id(aid: int) -> GameAction:
    a = _BY_ID.get(int(aid))
    if a is None:
        raise ValueError(f"no GameAction with id {aid!r}")
    return a


# --------------------------------------------------------------------------
# Grid helpers (pure functions, no framework dependency)
# Everything between the PURE markers is importable without arcengine, which
# is how scripts/test_agent.py exercises it in isolation.
# --- BEGIN PURE ---
# --------------------------------------------------------------------------

def settled_grid(frame: Any) -> Optional[list]:
    """Return the final grid from a frame response.

    FrameData.frame is list[list[list[int]]] -- a sequence of grids. Most
    steps return 2; an animation can return 20+. Only the last is the state
    the game settled into.
    """
    if not isinstance(frame, list) or not frame:
        return None
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return frame[-1]
    return frame


def grid_key(grid: Optional[list]) -> str:
    if not grid:
        return "empty"
    h = hashlib.blake2b(digest_size=8)
    for row in grid:
        h.update(bytes(bytearray(v & 0xFF for v in row)))
    return h.hexdigest()


def detect_scale(grid: list, candidates=(8, 4, 2, 1)) -> int:
    """Find the rendering block size.

    Games draw a small logical board scaled up into the 64x64 grid. If every
    kxk block is a solid colour, the logical cell size is k. Detecting this
    shrinks the click space by k^2 -- at k=4 that is 4096 -> 256 before we
    even filter by content.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for k in candidates:
        if k == 1 or rows % k or cols % k:
            continue
        ok = True
        for r0 in range(0, rows, k):
            if not ok:
                break
            for c0 in range(0, cols, k):
                v = grid[r0][c0]
                for r in range(r0, r0 + k):
                    if not ok:
                        break
                    for c in range(c0, c0 + k):
                        if grid[r][c] != v:
                            ok = False
                            break
                if not ok:
                    break
        if ok:
            return k
    return 1


def downsample(grid: list, k: int) -> list:
    if k <= 1:
        return grid
    return [[grid[r][c] for c in range(0, len(grid[0]), k)]
            for r in range(0, len(grid), k)]


def background_colour(small: list) -> int:
    counts: Counter = Counter()
    for row in small:
        counts.update(row)
    return counts.most_common(1)[0][0] if counts else 0


def components(small: list, bg: int, max_comps: int = 64) -> list:
    """4-connected same-colour regions of non-background cells.

    Returns (centre_row, centre_col, colour, size) per component, at the
    downsampled resolution.
    """
    rows, cols = len(small), len(small[0]) if small else 0
    seen = [[False] * cols for _ in range(rows)]
    out = []
    for r in range(rows):
        for c in range(cols):
            if seen[r][c] or small[r][c] == bg:
                continue
            colour = small[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not seen[nr][nc] and small[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            sr = sum(p[0] for p in cells) // len(cells)
            sc = sum(p[1] for p in cells) // len(cells)
            out.append((sr, sc, colour, len(cells)))
            if len(out) >= max_comps:
                return out
    return out


def click_candidates(grid: list, limit: int = 48) -> list:
    """Small set of (x, y) pixel coordinates worth clicking.

    Object centres plus their orthogonal neighbours -- sources and likely
    destinations. Reduces ~4096 blind options to a few dozen meaningful
    ones. Ordered smallest-object-first, since small distinct objects are
    usually the interactive pieces rather than background scenery.
    """
    if not grid:
        return []
    k = detect_scale(grid)
    small = downsample(grid, k)
    bg = background_colour(small)
    comps = components(small, bg)
    comps.sort(key=lambda t: t[3])

    rows, cols = len(small), len(small[0]) if small else 0
    seen: set = set()
    out: list = []

    def push(sr: int, sc: int) -> None:
        if not (0 <= sr < rows and 0 <= sc < cols):
            return
        # centre of that block in full-resolution pixel coords
        x = min(63, sc * k + k // 2)
        y = min(63, sr * k + k // 2)
        if (x, y) in seen:
            return
        seen.add((x, y))
        out.append((x, y))

    for sr, sc, _colour, _size in comps:
        push(sr, sc)
    for sr, sc, _colour, _size in comps:
        if len(out) >= limit:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            push(sr + dr, sc + dc)
    return out[:limit]


# --- END PURE ---

# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class MyAgent(Agent):
    """Novelty-seeking search over a graph of settled states."""

    # Total actions per game. The framework may cap lower.
    MAX_ACTIONS = 600
    # Give up after this many deaths that produced no new level.
    MAX_BARREN_DEATHS = 6
    # Click candidates considered per state.
    CLICK_LIMIT = 48

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(int(time.time() * 1_000_000) % 2**31)

        # Per-level: discarded when a level is cleared, since state keys
        # from the old level are meaningless in the new one.
        self.tried: dict[str, set] = defaultdict(set)
        self.edges: dict[tuple, str] = {}
        self.visits: Counter = Counter()

        # Per-game: survives level transitions. This is the memory that
        # makes later levels cheaper than the first.
        self.action_tried: Counter = Counter()
        self.action_changed: Counter = Counter()
        self.useful_clicks: Counter = Counter()

        # Bookkeeping
        self.prev_key: Optional[str] = None
        self.prev_choice: Optional[tuple] = None
        self.level_seen = 0
        self.deaths_since_progress = 0
        self.episode_start = 0
        self.episode_lengths: list[int] = []
        self.give_up = False

    @property
    def name(self) -> str:
        return f"{super().name}.graph"

    # -- helpers -----------------------------------------------------------

    def _legal(self, latest_frame: FrameData) -> list[int]:
        ids = [int(a) for a in (latest_frame.available_actions or [])
               if int(a) != 0]
        if not ids:
            ids = [a.value for a in GameAction if a is not GameAction.RESET]
        return ids

    def _expand(self, ids: list[int], grid: Optional[list]) -> list[tuple]:
        """Turn action ids into concrete choices.

        A choice is (action_id, x, y); x/y are None for simple actions.
        ACTION6 expands into one choice per click candidate.
        """
        out: list[tuple] = []
        for aid in ids:
            if aid == GameAction.ACTION6.value:
                for (x, y) in click_candidates(grid, self.CLICK_LIMIT):
                    out.append((aid, x, y))
            else:
                out.append((aid, None, None))
        return out

    def _prior(self, choice: tuple) -> float:
        """Historical usefulness of an action, learned across levels."""
        aid, x, y = choice
        tried = self.action_tried[aid]
        changed = self.action_changed[aid]
        base = (changed + 1.0) / (tried + 2.0)     # Laplace-smoothed
        if x is not None:
            base += 0.25 * min(self.useful_clicks[(x, y)], 4)
        return base

    def _build(self, choice: tuple, why: str) -> GameAction:
        aid, x, y = choice
        action = action_by_id(aid)
        if action.is_complex():
            action.set_data({"x": int(x or 0), "y": int(y or 0)})
            action.reasoning = {"why": why, "x": x, "y": y}
        else:
            action.reasoning = why
        return action

    def _record(self, key: str, grid_changed: bool) -> None:
        """Attribute the outcome of the previous action."""
        if self.prev_choice is None or self.prev_key is None:
            return
        aid, x, y = self.prev_choice
        self.action_tried[aid] += 1
        if grid_changed:
            self.action_changed[aid] += 1
            if x is not None:
                self.useful_clicks[(x, y)] += 1
        self.edges[(self.prev_key, self.prev_choice)] = key

    def _reset_level_memory(self) -> None:
        self.tried.clear()
        self.edges.clear()
        self.visits.clear()
        self.prev_key = None
        self.prev_choice = None

    # -- framework interface -----------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self.give_up:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        grid = settled_grid(latest_frame.frame)
        key = grid_key(grid)

        # A cleared level invalidates every state key we hold, but the
        # learned action statistics stay valid -- mechanics carry over.
        if latest_frame.levels_completed > self.level_seen:
            self.level_seen = latest_frame.levels_completed
            self.deaths_since_progress = 0
            self._reset_level_memory()

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self.episode_lengths.append(
                    self.action_counter - self.episode_start)
                self.deaths_since_progress += 1
                if self.deaths_since_progress >= self.MAX_BARREN_DEATHS:
                    # Repeated death with no new level. Stop and leave
                    # wall-clock for other games rather than grinding.
                    self.give_up = True
            self.episode_start = self.action_counter
            self.prev_key = None
            self.prev_choice = None
            return self._build((GameAction.RESET.value, None, None),
                               "reset")

        self._record(key, grid_changed=(key != self.prev_key))
        self.visits[key] += 1

        choices = self._expand(self._legal(latest_frame), grid)
        if not choices:
            return self._build((GameAction.RESET.value, None, None),
                               "no legal actions")

        done = self.tried[key]
        fresh = [c for c in choices if c not in done]

        if fresh:
            # Untried from here. Order by learned usefulness, with a little
            # noise so identical priors don't lock us into one ordering.
            fresh.sort(key=lambda c: -(self._prior(c) + random.random() * 0.1))
            choice = fresh[0]
            why = "unexplored"
        else:
            # Everything tried here. Head for the least-visited successor,
            # skipping self-loops, which are known no-ops.
            scored = []
            for c in choices:
                nxt = self.edges.get((key, c))
                if nxt == key:
                    continue
                scored.append((self.visits.get(nxt, 0), c))
            if scored:
                scored.sort(key=lambda t: (t[0], random.random()))
                choice = scored[0][1]
                why = "least-visited successor"
            else:
                choice = random.choice(choices)
                why = "all self-loops, random"

        self.tried[key].add(choice)
        self.prev_key = key
        self.prev_choice = choice
        return self._build(choice, why)
