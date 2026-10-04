# Live night runbook

## Fresh clone

Checked against the repo 2026-10-04 (nothing was run). There is no requirements.txt or pyproject.toml.
1. Python 3.11 env with torch (CPU) and `flyvis` (pins `<3.13`): `source ~/.venvs/ai/bin/activate`. Also needs `xdotool`, `gst-launch-1.0` + PipeWire, xdg-desktop-portal.
2. Git-ignored, so absent after `git clone` (`.gitignore`: `data/`, `logs/`, `calibration.json`, `calibration.json.bak`, `replay/`, `.portal_token`, `.env`, `specs/`, `*.md` except README/RUNBOOK):
   - `calibration.json`: `cp calibration.example.json calibration.json`, then `python calibrate.py` for your window size.
   - `data/` (flyvis connectome/results, `data/templates/*.npz` incl. `fly_readout.npz`, `readout_policy.npz`; corpus, features): not in git. `main.py` exits with "no fly readout" without `fly_readout.npz`; rebuilding it needs the labeled corpus (`python fly_brain.py`), which is also not in git. Copy `data/` from the original machine, or use `--perception pixel` for a no-fly fallback.
   - `logs/` is created at run time; `touch logs/STOP` needs the directory to exist (`mkdir -p logs`).
3. `pytest -q`: offline tests; the slow replay tests need `data/corpus` and the fitted readout, so they fail or skip on a bare clone.
4. Then the steps below.

Before: game running in the `FNAF` Bottles bottle, in front, fullscreen. `calibration.json` matches this window size
(`python calibrate.py` if the window moved or resized). Hands off the mouse while the bot runs (a touch over the bottom bar
raises the monitor).

1. `python main.py --night 1 --dry-run --max-seconds 60` check: reads clock/power/doors, logs a row per tick, never clicks.
2. `python main.py --night 1 --viz` start on the Night 1 title card (guards allow 25 s before the first trusted state).
   Open http://localhost:8765 to watch the fly's neurons and the danger score per hallway.
3. Stop with `touch logs/STOP`. SAFE_MODE (state unreadable 5 s, too many actions/min, stuck monitor) stops clicking by itself.
4. After: `python evaluate.py` for survival stats; `logs/run_*.jsonl` has timestamp, state, danger, action and reason per tick.

Known behaviour: the monitor only toggles on pointer motion into the bottom bar (the actuator glides in); light and door
clicks are toggles and are never retried; with a door closed the fly sees the slab, so the door reopens blind after 12 s and
the next light check decides. Foxy and Chica are not handled yet (phases 15/26).

Not applied automatically: ES-tuned timings (`logs/es_proposal.json`). Try them with `python fit_config.py KEY=value ...`.
