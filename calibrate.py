"""Phase 10: walk REQUIRED_BUTTONS, hover the pointer over each, write calibration.json (reference coords)."""
import argparse
import json
import subprocess
from pathlib import Path

import config
from geometry import REQUIRED_BUTTONS, Geometry, Window, in_window, window_to_ref

DEFAULT_SIZE = (40, 40)  # reference px for single-hover buttons (doors, lights, cams)
RECT_BUTTONS = {"monitor_bar", "hall_L", "hall_R", "clock", "power", "usage", "cam_map"}  # two corner hovers


def build_calibration(samples, geom, base=None):
    """samples: {name: [(px, py)] window-relative px; 1 point (centre) or 2 (opposite corners)} -> dict."""
    buttons = dict((base or {}).get("buttons", {}))
    for name, pts in samples.items():
        for p in pts:
            if not in_window(geom, *p, margin=0):
                raise ValueError(f"{name}: pointer {p} outside the {geom.window.w}x{geom.window.h} game window")
        ref = [window_to_ref(geom, *p) for p in pts]
        if len(ref) == 1:
            (cx, cy), (w, h) = ref[0], DEFAULT_SIZE
            rect = (cx - w / 2, cy - h / 2, w, h)
        else:
            (x0, y0), (x1, y1) = ref[:2]
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        buttons[name] = [round(v) for v in rect]
    return {"uncalibrated": any(n not in buttons for n in REQUIRED_BUTTONS),
            "ref": [config.REF_W, config.REF_H], "buttons": buttons}


def _pointer_in_window(win):
    out = subprocess.run(["xdotool", "getmouselocation", "--shell"], capture_output=True, text=True).stdout
    m = dict(l.split("=") for l in out.split())
    return int(m["X"]) - win[1], int(m["Y"]) - win[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--path", default=config.CALIBRATION_PATH)
    a = ap.parse_args(argv)
    from capture import SCALE, find_window
    win = find_window()
    if win[0] is None:
        raise SystemExit("game window not found")
    geom = Geometry(Window(*win), SCALE)
    base = json.loads(Path(a.path).read_text()) if a.only and Path(a.path).exists() else None
    samples = {}
    for name in [a.only] if a.only else REQUIRED_BUTTONS:
        pts = []
        for corner in (["top-left", "bottom-right"] if name in RECT_BUTTONS else [""]):
            while True:
                input(f"hover the pointer over {name} {corner} and press Enter ")
                p = _pointer_in_window(win)
                if in_window(geom, *p, margin=0):
                    pts.append(p)
                    break
                print("pointer is outside the game window, try again")
        samples[name] = pts
    cal = build_calibration(samples, geom, base)
    Path(a.path).write_text(json.dumps(cal, indent=1))
    print(f"wrote {a.path} (uncalibrated={cal['uncalibrated']})")


if __name__ == "__main__":
    main()
