"""Phase 04: mouse actuator for FNaF 1. Every xdotool call targets self.wid only.

Proton ignores XSendEvent, so never `click --window`: activate, mousemove --window, then
mousedown/mouseup. Timings and LIGHT_MODE are unverified until `python actuator.py --probe`.
"""
import subprocess
import sys
import time
from contextlib import contextmanager

import config
from actions import Action, Decision
from geometry import center, in_window, load_buttons, ref_to_window


class Refused(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Actuator:
    def __init__(self, wid, geom, buttons, run=subprocess.run, sleep=time.sleep, clock=time.monotonic):
        # Never fall through to "whatever has focus": that could be a terminal.
        if wid is None:
            raise RuntimeError("FNaF window not found. Launch the game first.")
        self.wid, self.geom, self.buttons = str(wid), geom, buttons
        self._run, self._sleep, self._clock = run, sleep, clock
        self._last = None

    def _xdo(self, *args):
        return self._run(["xdotool", *args], capture_output=True, text=True)

    def _point(self, rx, ry):
        px, py = ref_to_window(self.geom, rx, ry)
        if not in_window(self.geom, px, py):
            raise Refused(f"({rx},{ry}) -> ({px},{py}) outside {self.geom.window.w}x{self.geom.window.h} window")
        return str(px), str(py)

    def _move(self, rx, ry):
        self._xdo("mousemove", "--window", self.wid, *self._point(rx, ry))

    def _goto(self, name):
        pt = self._point(*center(self.buttons[name]))  # validate before any xdotool call
        self._xdo("windowactivate", "--sync", self.wid)
        self._xdo("mousemove", "--window", self.wid, *pt)

    def click(self, name):
        self._goto(name)
        self._sleep(config.PAN_SETTLE)
        self._xdo("mousedown", "1")
        self._sleep(config.CLICK_HOLD)
        self._xdo("mouseup", "1")

    def hover(self, name, dwell):
        self._goto(name)
        self._sleep(dwell)

    @contextmanager
    def light_hold(self, side):
        self._goto(f"light_{side}")
        self._sleep(config.PAN_SETTLE)
        self._xdo("mousedown", "1")
        try:
            yield
        finally:
            self._xdo("mouseup", "1")

    def light(self, side, on=True):
        if config.LIGHT_MODE == "toggle":
            self.click(f"light_{side}")  # caller tracks state
        elif on:
            with self.light_hold(side):
                self._sleep(config.LIGHT_HOLD)

    def monitor(self):
        # The game only toggles on pointer *motion* into the bar; a single warp raised it but never lowered it (live test),
        # so glide in from above in small steps.
        pt = self._point(*center(self.buttons["monitor_bar"]))  # validate before any xdotool call
        self._xdo("windowactivate", "--sync", self.wid)
        cx, cy = center(self.buttons["monitor_bar"])
        for dy in range(-120, 1, 15):
            self._move(cx, cy + dy)
            self._sleep(0.05)
        self._sleep(config.MONITOR_DWELL)

    def cam(self, cam_id):
        self.click(f"cam_{cam_id}")

    def park(self):
        self._move(*config.PARK_REF)

    def act(self, d: Decision):
        """-> True if something was sent."""
        if d.action == Action.NONE:
            return False
        now = self._clock()
        if self._last is not None and now - self._last < config.MIN_ACTION_GAP:
            # wait out the gap rather than drop: the supervisor already counted this action as sent (live bug: back-to-back
            # LIGHT_L / LIGHT_R lost the second one)
            self._sleep(config.MIN_ACTION_GAP - (now - self._last))
        a = d.action
        if a == Action.SAFE_MODE:
            self.park()
        elif a in (Action.DOOR_L, Action.DOOR_R):
            self.click("door_" + a.name[-1])
        elif a in (Action.LIGHT_L, Action.LIGHT_R):
            self.light(a.name[-1])
        elif a == Action.MONITOR:
            self.monitor()
        elif a == Action.CAM:
            self.cam(d.arg)
        self._last = self._clock()  # gap counts from the end of the action
        return True


PROBE_QUESTIONS = """\
Look at the screenshots in {out}:
F1/F4: did each door click close/open the door, and does hovering the bottom bar raise the monitor?
F5: light_L once -> did the light stay on (toggle) or only while held? (set config.LIGHT_MODE)
F6: did the light work with the door closed?
Record answers in specs/docs/fnaf1-facts.md.
"""


def probe(calib_path=config.CALIBRATION_PATH):
    buttons = load_buttons(calib_path)  # raises Uncalibrated before anything touches the game
    from pathlib import Path

    from PIL import Image

    from capture import SCALE, Capture, find_window
    from geometry import Geometry, Window

    cap = Capture()
    act = Actuator(cap.wid, Geometry(Window(*find_window()), SCALE), buttons)
    out = Path(config.LOG_DIR) / "probe" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    n = 0

    def shot(label):
        nonlocal n
        time.sleep(1.0)
        Image.fromarray(cap.grab()).save(out / f"{n:02d}_{label}.png")
        n += 1

    act.park()
    shot("parked")
    steps = [("door_L", act.click), ("door_L", act.click), ("door_R", act.click), ("door_R", act.click)]
    for name, fn in steps:
        fn(name)
        shot(f"click_{name}")
    for side in "LR":
        act.light(side)
        shot(f"light_{side}_1")
        if side == "L":
            act.light(side)
            shot(f"light_{side}_2")
    act.monitor()
    shot("monitor_1")
    act.monitor()
    shot("monitor_2")
    act.park()
    cap.close()
    (out / "probe.txt").write_text(PROBE_QUESTIONS.format(out=out))
    print(PROBE_QUESTIONS.format(out=out))


if __name__ == "__main__":
    if "--probe" not in sys.argv:
        sys.exit("usage: python actuator.py --probe")
    probe()
