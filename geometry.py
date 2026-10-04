"""Coordinate math: reference (1280x720) <-> window px (xdotool) <-> capture-frame px."""
import json
from dataclasses import dataclass

from config import REF_H, REF_W

Rect = tuple  # (rx, ry, rw, rh) in reference px

REQUIRED_BUTTONS = ["door_L", "door_R", "light_L", "light_R", "monitor_bar", "hall_L", "hall_R",
                    "clock", "power", "usage", "cam_map",
                    "cam_1A", "cam_1B", "cam_1C", "cam_2A", "cam_2B", "cam_3", "cam_4A", "cam_4B",
                    "cam_5", "cam_6", "cam_7"]


class Uncalibrated(Exception):
    pass


@dataclass
class Window:
    wid: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class Geometry:
    window: Window
    capture_scale: int = 2


def ref_to_window(geom, rx, ry):
    # no clamping: out-of-range must stay detectable via in_window
    return round(rx * geom.window.w / REF_W), round(ry * geom.window.h / REF_H)


def window_to_capture(geom, px, py):
    return px // geom.capture_scale, py // geom.capture_scale


def capture_to_ref(geom, cx, cy):
    s = geom.capture_scale  # +0.5: pixel centre, halves the floor error of window_to_capture
    return (cx + 0.5) * s * REF_W / geom.window.w, (cy + 0.5) * s * REF_H / geom.window.h


def in_window(geom, px, py, margin=2):
    return margin <= px < geom.window.w - margin and margin <= py < geom.window.h - margin


def center(rect):
    rx, ry, rw, rh = rect
    return rx + rw / 2, ry + rh / 2


def load_buttons(path):
    d = json.load(open(path))
    b = d.get("buttons", {})
    if d.get("uncalibrated") or any(n not in b for n in REQUIRED_BUTTONS):
        raise Uncalibrated(f"{path}: uncalibrated or missing buttons (run Phase 11 calibration)")
    return {k: tuple(v) for k, v in b.items()}


def roi_box(frame, geom, name, buttons):
    """-> (x0, y0, x1, y1) of a button/ROI in capture-frame pixels, clipped to the frame."""
    rx, ry, rw, rh = buttons[name]
    x0, y0 = window_to_capture(geom, *ref_to_window(geom, rx, ry))
    x1, y1 = window_to_capture(geom, *ref_to_window(geom, rx + rw, ry + rh))
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"roi {name} empty after clipping to {w}x{h} frame")
    return x0, y0, x1, y1


def roi(frame, geom, name, buttons):
    x0, y0, x1, y1 = roi_box(frame, geom, name, buttons)
    return frame[y0:y1, x0:x1]


def window_to_ref(geom, px, py):
    return px * REF_W / geom.window.w, py * REF_H / geom.window.h
