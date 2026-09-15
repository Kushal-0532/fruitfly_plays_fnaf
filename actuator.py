"""Phase 05: send policy actions to the UCN window with xdotool.

CONFIRMED KEYBINDINGS (read from UCN's in-game keybind screen, 2026-09-14):
    A      Close Left Door   (toggle)
    S      Monitor           (toggle)
    D      Close Right Door  (toggle)
    Z      Flashlight        W  Close Forward Vent   F  Close Side Vent
    Space  Desk Fan          X/6  All OFF            Hold Esc  Return to Menu
Camera selection is mouse-only (no key).
Doors can be toggled while the monitor is up (user-confirmed).
"""
import subprocess
import time

from policy import CAM_TOGGLE, LEFT_DOOR, NONE, RIGHT_DOOR

KEYS = {CAM_TOGGLE: "s", LEFT_DOOR: "a", RIGHT_DOOR: "d"}
KEY_HOLD = 0.06  # s. Games polling key state at 60 fps can miss an instant tap.
# UCN pans the office view with the pointer, and hovering the bottom bars opens
# the monitor. Park the pointer so the LEFT door (Freddy + Foxy) stays in view,
# well above the bottom bars. Fractions of the game window. Calibration knob.
PARK = (0.08, 0.45)


def _xdo(*args):
    return subprocess.run(["xdotool", *args], capture_output=True, text=True)


class Actuator:
    def __init__(self, wid):
        # Never fall through to "whatever has focus": that could be a terminal.
        if wid is None:
            raise RuntimeError("UCN window not found. Launch the game first.")
        self.wid = str(wid)

    def park(self):
        g = dict(l.split("=") for l in _xdo("getwindowgeometry", "--shell", self.wid).stdout.split())
        x, y = int(int(g["WIDTH"]) * PARK[0]), int(int(g["HEIGHT"]) * PARK[1])
        _xdo("mousemove", "--window", self.wid, str(x), str(y))

    def act(self, action):
        if action == NONE:
            return
        key = KEYS[action]
        # ponytail: activate on every press. Cheap and survives focus theft.
        _xdo("windowactivate", "--sync", self.wid)
        _xdo("keydown", key)
        time.sleep(KEY_HOLD)
        _xdo("keyup", key)


if __name__ == "__main__":
    from pathlib import Path

    from PIL import Image

    from capture import Capture

    cap = Capture()
    act = Actuator(cap.wid)
    Path("logs/act").mkdir(parents=True, exist_ok=True)
    act.park()
    time.sleep(1.5)  # let the view finish panning
    Image.fromarray(cap.grab()).save("logs/act/0_parked.png")
    for i, (a, label) in enumerate([(CAM_TOGGLE, "monitor_up"), (CAM_TOGGLE, "monitor_down"),
                                     (LEFT_DOOR, "left_closed"), (LEFT_DOOR, "left_open"),
                                     (RIGHT_DOOR, "right_closed"), (RIGHT_DOOR, "right_open")], 1):
        print("press", KEYS[a], "->", label, flush=True)
        act.act(a)
        time.sleep(2.0)  # monitor flip animation needs time
        Image.fromarray(cap.grab()).save(f"logs/act/{i}_{label}.png")
    cap.close()
    print("saved logs/act/*.png")
