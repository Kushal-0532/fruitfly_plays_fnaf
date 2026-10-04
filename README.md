# Fruit fly plays Five Nights at Freddy's 1

A bot that plays FNaF 1 where **the fly decides**: the game screen goes into a connectome-constrained model of the fruit fly
visual system, a few of its neurons over the hallway patches form a danger score, and that score closes the door. Plain code
only does bookkeeping (when to look, timers, safety guards, clicks).

```
capture.py -> readers (clock, power, doors, lights) -> fly_brain.py (flyvis, danger per hallway) -> policy.py (Supervisor)
          -> safety.py (Guards) -> actuator.py (xdotool clicks) ;  main.py wires the loop, every tick logged to logs/run_*.jsonl
```

## Run
```
source ~/.venvs/ai/bin/activate
pytest -q                                   # offline tests (the slow replay tests need data/corpus + the fitted readout)
python main.py --night 1 --viz              # live; fly neurons at http://localhost:8765
python main.py --night 1 --dry-run          # read + log only, never clicks
python main.py --night 1 --perception pixel # old pixel detector (comparison/fallback only)
touch logs/STOP                             # stop
```
See [RUNBOOK.md](RUNBOOK.md) for a live night. Linux/GNOME Wayland: screen capture goes through the xdg-desktop-portal
ScreenCast + PipeWire; clicks go to the Xwayland game window with xdotool.

## Offline tools
- `python fly_brain.py` refit the danger readout from the labeled corpus
- `python es_tune.py` evolution strategy over policy timings in a coarse simulated night (`sim_env.py`); writes a proposal to
  `logs/es_proposal.json`, apply with `python fit_config.py KEY=value` only after a live check
- `python -m scripts.replay_run data/corpus/n2_a 3640 3700` closed-loop replay of recorded frames (dry run)
- `python export_replay.py ...` offline replay HTML of the fly's neurons; `python evaluate.py` run statistics

## Credits
- [`flyvis`](https://github.com/TuragaLab/flyvis) (TuragaLab): Lappalainen et al. 2024, *Connectome-constrained networks
  predict neural activity across the fly visual system*, Nature.
- Five Nights at Freddy's is by Scott Cawthon; this is an unaffiliated hobby project.
