"""Phase 13: door / light / monitor / camera state from colour rules on the button ROIs.
Door button is red (open) or green (closed); light button turns blue when on; the camera map's white
lines only exist in the monitor view; the selected camera's map box is yellow-green (and it blinks)."""
import numpy as np

from geometry import REQUIRED_BUTTONS, roi
from state import Readings

CAMS = [n[4:] for n in REQUIRED_BUTTONS if n.startswith("cam_") and n != "cam_map"]
MIN_FRAC = 0.2      # share of a ROI that must be the state colour
MAP_FRAC = 0.02     # bright share of the cam_map ROI above which the monitor is up (measured: down <= 0.002, up >= 0.045)
CAM_FRAC, CAM_GAP = 0.15, 0.1


def _frac(r, cond):
    r = r.astype(int)
    return float(cond(r[..., 0], r[..., 1], r[..., 2]).mean())


class ButtonReader:
    def __init__(self, geom, buttons):
        self.geom, self.buttons = geom, buttons

    def _roi(self, frame, name):
        return roi(frame, self.geom, name, self.buttons)

    def monitor_up(self, frame):
        return _frac(self._roi(frame, "cam_map"), lambda R, G, B: (R + G + B) / 3 > 200) > MAP_FRAC

    def door_closed(self, frame, side):
        r = self._roi(frame, "door_" + side)
        if _frac(r, lambda R, G, B: (G > 150) & (R < 100)) > MIN_FRAC:
            return True
        return False if _frac(r, lambda R, G, B: (R > 150) & (G < 80)) > MIN_FRAC else None  # None: button off-screen

    def light_on(self, frame, side):
        if self.door_closed(frame, side) is None:  # the light button sits under the door button; no door button = not visible
            return None
        return _frac(self._roi(frame, "light_" + side), lambda R, G, B: (B > 150) & (B > R + 30)) > MIN_FRAC

    def cam(self, frame):
        sc = {c: _frac(self._roi(frame, "cam_" + c), lambda R, G, B: (G > 150) & (R > 100) & (B < 90) & (G > B + 80)) for c in CAMS}
        (c1, s1), (_, s2) = sorted(sc.items(), key=lambda x: -x[1])[:2]
        return c1 if s1 > CAM_FRAC and s1 - s2 > CAM_GAP else None

    def read(self, frame, t):
        up = self.monitor_up(frame)
        if up:  # office buttons are covered by the camera view
            return Readings(t, monitor_up=True, cam=self.cam(frame))
        return Readings(t, monitor_up=False,
                        door_closed={s: self.door_closed(frame, s) for s in "LR"},
                        light_on={s: self.light_on(frame, s) for s in "LR"})
