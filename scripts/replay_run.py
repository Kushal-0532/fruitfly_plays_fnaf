"""Closed-loop replay: recorded corpus frames play the part of the camera, everything else is the real stack
(readers, the fly in its thread, tracker, supervisor, guards, main.run) in real time. Decisions are logged, nothing is clicked.
usage: python scripts/replay_run.py data/corpus/n2_a 3640 3720"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
from fly_brain import FlyBrain, FlyHallway
from geometry import load_buttons
from main import run
from policy import Supervisor
from power import PowerModel
from readers import live_reader
from readers.digits import _geom
from safety import Guards
from state import StateTracker


class ReplayCapture:
    def __init__(self, session, f0, f1, fps=10):
        self.d, self.f0, self.f1, self.fps = Path(session), f0, f1, fps
        self.t0, self.n, self._i = None, 0, None
        self.cache = {}

    def start(self):
        self.t0 = time.monotonic()

    def grab(self):
        i = self.f0 + int((time.monotonic() - self.t0) * self.fps)
        if i >= self.f1:
            raise RuntimeError("replay over")
        if i != self._i:
            self._i, self.n = i, self.n + 1
        if i not in self.cache:
            self.cache[i] = np.asarray(Image.open(self.d / "frames" / f"{i:06d}.png").convert("RGB"))
        return self.cache[i]


class NoActuator:
    def act(self, d):
        raise AssertionError("dry run must not act")


def replay(session, f0, f1, out="logs/replay_run.jsonl"):
    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    geom, buttons = _geom(meta), load_buttons(config.CALIBRATION_PATH)
    brain = FlyBrain(geom, buttons, view=False)
    cap = ReplayCapture(session, f0, f1)
    brain.submit(np.asarray(Image.open(d / "frames" / f"{f0:06d}.png").convert("RGB")), 0)
    while brain.snapshot() is None:  # the network takes a few seconds to load
        if brain.error is not None:
            raise RuntimeError(f"fly brain thread died: {brain.error!r}")
        time.sleep(0.2)
    cap.start()
    cap.cache.clear()
    summary = run(cap, live_reader(geom, buttons, FlyHallway(brain)), NoActuator(), StateTracker(assume_open=True),
                  Supervisor(PowerModel()), Guards(stop_path=Path(out).with_suffix(".STOP")), out, 2, dry_run=True,
                  max_seconds=(f1 - f0) / 10 + 1)
    return [json.loads(l) for l in open(out)][1:]


if __name__ == "__main__":
    rows = replay(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    for r in rows:
        if r["action"] != "NONE" or any(v and v > 0.5 for v in r["hall"].values() if v is not None):
            print(round(r["t"], 1), r["action"], r["reason"], r["hall"], r["door_closed"], r["light_on"])
