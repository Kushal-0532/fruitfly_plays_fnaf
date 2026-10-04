# Live night runbook

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
