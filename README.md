# ARC Prize 2026 — Local Dev Kit

**Go from zero to your first Kaggle submission in about ten minutes, without ever opening
the Kaggle notebook editor.**

A development kit for the
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. You edit one Python file on your laptop, watch it play the real game
environments locally, and push it to Kaggle as a submission with a single command.

No Docker. No `submission.json` to hand-write. No copy-pasting between your editor and a
notebook.

---

## What you need before you start

- **Python 3.12** — the competition's `arc-agi` package requires it
  - macOS: `brew install python@3.12`
  - Ubuntu: `sudo apt install python3.12 python3.12-venv`
  - Windows: install from [python.org](https://www.python.org/downloads/)
- **git**, to clone the official agent framework during setup
- **A Kaggle account** with the competition rules accepted
  ([accept here](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules))

No GPU required for the starter agent.

---

## Quick start

```bash
# 1. Clone this repo and step in
git clone https://github.com/Rohan199/ARC-AGI-3-starter.git
cd ARC-AGI-3-starter

# 2. Drop your Kaggle API token (kaggle.com -> Settings -> Create New Token)
#    into the project-local .kaggle/ folder, NOT your home directory
mkdir -p .kaggle && echo "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .kaggle/access_token
chmod 600 .kaggle/access_token

# 3. One-time setup: venv, dependencies, framework
make setup

# 4. Open agent/my_agent.py to see the random-action starter, then edit it.
#    This is the only file you change.

# 5. Run it locally against every game in the competition (takes seconds)
make play-local

# 6. Push it to Kaggle as a submission notebook
make submit

# 7. Watch the run
make status

# 8. When status shows "complete", open the notebook on kaggle.com, find your
#    kernel, click "Submit to Competition" in the top right, and pick
#    submission.parquet from the Output File dropdown.
#    That spends one of your five daily submissions.
```

Steps 4–7 are the loop you repeat while iterating. Step 8 is the deliberate moment when
you spend a submission.

---

## The one file you edit: `agent/my_agent.py`

It defines a class called `MyAgent` with two methods:

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """Return True when your agent wants to stop playing."""
        ...

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Look at the game state and return the next action."""
        ...
```

The starter version picks random actions — a baseline that proves the pipeline works end
to end. Replace the body of `choose_action` with your strategy. Kaggle plumbing,
submission file format, and game orchestration are handled for you.

---

## What `make submit` does

ARC-AGI-3 is a code competition: you submit a notebook and Kaggle runs it twice.

`make submit` builds and uploads the notebook (Phase A). Once `make status` reports
`complete`, open the kernel on kaggle.com and click **Submit to Competition** to enter
Phase B and get a leaderboard score.

> **Five official submissions per day.** Get `make play-local` passing before you spend one.

> **Before your first submit,** open `notebooks/kernel-metadata.json` and replace
> `REPLACE_WITH_YOUR_USERNAME` with your Kaggle handle. The Makefile refuses to push
> until you do.

### Choosing an accelerator

The notebook is generated with a **T4 GPU** by default, matching Kaggle's sample
submission. To change it, edit one line near the top of `scripts/build_notebook.py`:

```python
ACCELERATOR = "t4"     # one of: cpu, t4, p100, rtx6000
```

Re-run `make submit` and both the notebook metadata and `notebooks/kernel-metadata.json`
update automatically.

| Value       | Hardware                           | When to use                                              |
| ----------- | ---------------------------------- | -------------------------------------------------------- |
| `"cpu"`     | No GPU                             | The random starter, or any non-ML agent                  |
| `"t4"`      | Nvidia T4 x2                       | **Default.** Small models, fast iteration                |
| `"p100"`    | Nvidia P100                        | Single big-memory GPU                                    |
| `"rtx6000"` | Nvidia RTX 6000 (`g4-standard-48`) | Heavy ML; ARC-AGI-3 exclusive, burns GPU quota faster    |

RTX 6000 is reserved for ARC-AGI-3 notebooks — don't use it for early iteration. All
accelerated Kaggle sessions have internet disabled, which is already the default here.

---

## All the commands

| Command                     | What it does                                                                |
| --------------------------- | --------------------------------------------------------------------------- |
| `make setup`                | One-time install: venv, `arc-agi`, `kaggle` CLI, clones the framework       |
| `make play-local`           | Runs your agent against every game in the dataset, locally                  |
| `make play-local GAME=ls20` | Same, but one game only — faster while debugging                            |
| `make verify-local`         | 30-second smoke test on two games                                           |
| `make list-games`           | Print every available game id                                               |
| `make pull-sample`          | Download the official sample agent for reference                            |
| `make notebook`             | Build the Kaggle notebook from your agent, without pushing                  |
| `make submit`               | Build the notebook **and** push it to Kaggle                                |
| `make status`               | Check the status of your most recent Kaggle run                             |
| `make clean`                | Remove the venv, downloads, and generated notebook                          |

---

## Why not just edit in the Kaggle notebook?

1. **Iteration speed.** Your normal IDE plus `make play-local` gives a real-game-engine
   feedback loop in seconds. The Kaggle editor's loop is minutes per change.
2. **No environment surprises.** The local `arc-agi` PyPI package hosts the same game
   engine the Kaggle gateway runs. If it works locally, it works on Kaggle.
3. **Your code stays in git.** Notebooks are awful for diffs and review. Your real work
   lives in `agent/my_agent.py`; the notebook is an auto-generated deployment artifact.

---

## Project layout

```
.
├── agent/
│   └── my_agent.py             * The file you edit
├── scripts/
│   ├── play_local.py           Runs your agent against real games
│   ├── build_notebook.py       Packages your agent into a Kaggle notebook
│   └── slim_framework.py       Trims framework deps so install stays light
├── notebooks/
│   ├── kernel-metadata.json    Edit once: your Kaggle username
│   └── submission.ipynb        Auto-generated, never edit by hand
├── probes/                     Exploratory scripts for probing game behaviour
├── logs/                       Local run output
├── vendor/                     Cloned framework (gitignored)
├── .venv/                      Python 3.12 venv (gitignored)
├── .kaggle/                    Project-local Kaggle token (gitignored)
└── Makefile
```

---

## Troubleshooting

**`make setup` fails with `python3.12: command not found`**
Install Python 3.12 — the `arc-agi` package requires it. macOS: `brew install python@3.12`.

**`make submit` says "edit kernel-metadata.json"**
You haven't replaced `REPLACE_WITH_YOUR_USERNAME` in `notebooks/kernel-metadata.json`.

**`make submit` returns `401 Unauthorized`**
Your Kaggle token is missing or invalid. Generate a fresh one from your
[Kaggle settings](https://www.kaggle.com/settings) and overwrite `.kaggle/access_token`.

**`make play-local` says "Could not create environment"**
Your machine couldn't reach the ARC-AGI API to download the game source on first run.
Check your connection and retry — once downloaded, games are cached in
`environment_files/` and you're fully offline.

**My local score is 0.0**
Expected for the random starter agent. Making it non-zero is the job.

---

## Where to go next

- The [ARC-AGI-3 docs](https://docs.arcprize.org/) for the benchmark itself
- `make pull-sample` to study Kaggle's reference agent
- The competition's
  [discussion forum](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion)

---

## Acknowledgements

Built around the official ARC-AGI-3 Kaggle agent framework, which `make setup` clones into
`vendor/`. The local development workflow, notebook build pipeline, and Makefile tooling in
this repository are my own.

## License

MIT — see [LICENSE](LICENSE).
