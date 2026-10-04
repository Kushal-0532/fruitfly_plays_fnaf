"""Shared test fakes. No test may call real xdotool/portal."""
import subprocess

import numpy as np


class FakeRunner:
    """Callable stand-in for subprocess.run. `script` is a list of stdout strings (or
    CompletedProcess) consumed in order; the last one repeats. Every argv is recorded."""

    def __init__(self, script=("",)):
        self.script = list(script)
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(r, str):
            r = subprocess.CompletedProcess(argv, 0, stdout=r, stderr="")
        return r


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def synth_frame(w=640, h=360, **rects):
    """uint8 RGB frame; rects: name=(x0, y0, x1, y1, (r, g, b))."""
    f = np.zeros((h, w, 3), np.uint8)
    for x0, y0, x1, y1, rgb in rects.values():
        f[y0:y1, x0:x1] = rgb
    return f


def make_state(**kw):
    """Trusted quiet GameState: doors open, lights off, hallways empty, Foxy stage 0."""
    from state import GameState
    d = dict(t=0.0, hour=1, power_pct=95.0, usage=1, monitor_up=False, cam=None, foxy_stage=0,
             door_closed={"L": False, "R": False}, light_on={"L": False, "R": False},
             hall={"L": 0.0, "R": 0.0}, trusted=True, reasons=[])
    d.update(kw)
    return GameState(**d)


def night_quiet():
    """6 hours, nobody ever at a door."""
    return []


def night_bonnie_left_visit(t0=20.0, dur=12.0):
    """Left hallway occupied from t0 for dur seconds."""
    return [(t0, {"hall": {"L": 0.9, "R": 0.0}}), (t0 + dur, {"hall": {"L": 0.0, "R": 0.0}})]
