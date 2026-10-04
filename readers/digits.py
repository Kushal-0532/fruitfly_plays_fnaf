"""Phase 12: hour / power / usage readers. The HUD is tiny white-on-dark pixel font, so this is
plain template matching on binarised ROIs (numpy only). Every read returns None rather than guess."""
import csv
import json
from pathlib import Path

import numpy as np

from geometry import Geometry, Window, roi
from state import Readings

THRESH = 190          # gray level above which a pixel is "text"
CELL = (11, 9)        # glyph cell (rows, cols), top-left aligned
HOUR_COLS = 30        # hour digits live in the left part of the clock ROI (the "AM" is clipped)
MIN_SCORE = 0.85      # ponytail: hand-picked, not fitted; refit from the confusion data if reads go wrong
HOUR_MARGIN = 0.03    # best hour must beat the runner-up by this much
USAGE_SEGS = 4        # bar slots in the usage ROI, equal width
BAR_G = 170           # lit bar: green channel above this and blue below BAR_B (green and yellow bars; the red 4th bar is not counted)
BAR_B = 110
BAR_FRAC = 0.5


def _mask(r):
    return (r.mean(axis=2) if r.ndim == 3 else r) > THRESH


def _runs(mask):
    """-> [(x0, x1)] inclusive column runs that contain any text pixel."""
    cols, out, start = mask.any(axis=0), [], None
    for x, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = x
        elif not c and start is not None:
            out.append((start, x - 1))
            start = None
    return out


def _cell(mask, x0, x1):
    rows = np.flatnonzero(mask[:, x0:x1 + 1].any(axis=1))
    sub = mask[rows[0]:rows[-1] + 1, x0:x1 + 1][:CELL[0], :CELL[1]]
    c = np.zeros(CELL, bool)
    c[:sub.shape[0], :sub.shape[1]] = sub
    return c


def _score(a, b):
    if a.shape != b.shape:  # ROI clipped differently from the template: treat as no match
        return 0.0
    return 1.0 - float((a != b).mean())


def split_digits(r):
    """ROI -> list of glyph cells, left to right (the % sign comes out as an extra, non-matching cell)."""
    m = _mask(r)
    return [_cell(m, x0, x1) for x0, x1 in _runs(m)]


def _geom(meta):
    return Geometry(Window(**meta["window"]), meta["capture_scale"])


def _labeled(session_dirs, keep=lambda i: True):
    """yield (frame ndarray, geom, label row) for every labeled row of the sessions, sorted."""
    from PIL import Image
    for d in sorted(map(Path, session_dirs)):
        meta = json.loads((d / "meta.json").read_text())
        geom = _geom(meta)
        with open(d / "labels.csv", newline="") as f:
            for row in sorted(csv.DictReader(f), key=lambda r: int(r["frame"])):
                i = int(row["frame"])
                if keep(i):
                    yield np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB")), geom, row


def build_templates(session_dirs, buttons, out="data/templates/digits.npz", keep=lambda i: True):
    """Mean binary glyph per digit 0-9 (from power_pct labels) and per hour 0-6 (from hour labels)."""
    dig, hrs = {}, {}
    for frame, geom, row in _labeled(session_dirs, keep):
        if (row.get("power_pct") or "").strip():
            s = str(int(float(row["power_pct"])))
            cells = split_digits(roi(frame, geom, "power", buttons))
            if len(cells) == len(s) + 1:  # digits + the % glyph
                for ch, c in zip(s, cells):
                    dig.setdefault(int(ch), []).append(c)
        if (row.get("hour") or "").strip():
            hrs.setdefault(int(row["hour"]), []).append(_mask(roi(frame, geom, "clock", buttons))[:, :HOUR_COLS])
    arrs = {f"d{k}": np.mean(v, axis=0) > 0.5 for k, v in sorted(dig.items())}
    arrs.update({f"h{k}": np.mean(v, axis=0) > 0.5 for k, v in sorted(hrs.items())})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrs)  # deterministic: sorted insertion, no randomness
    return {k: int(v.sum()) for k, v in arrs.items()}


class DigitReader:
    def __init__(self, geom, buttons, templates_path="data/templates/digits.npz"):
        self.geom, self.buttons = geom, buttons
        z = np.load(templates_path)
        self.dig = {int(k[1:]): z[k] for k in z.files if k[0] == "d"}
        self.hrs = {int(k[1:]): z[k] for k in z.files if k[0] == "h"}

    def _roi(self, frame, name):
        return roi(frame, self.geom, name, self.buttons)

    def read_hour(self, frame):
        m = _mask(self._roi(frame, "clock"))[:, :HOUR_COLS]
        if not m.any() or not self.hrs:
            return None
        sc = sorted(((_score(m, t), h) for h, t in self.hrs.items()), reverse=True)
        second = sc[1][0] if len(sc) > 1 else 0.0
        return sc[0][1] if sc[0][0] >= MIN_SCORE and sc[0][0] - second >= HOUR_MARGIN else None

    def read_power(self, frame):
        cells, s = split_digits(self._roi(frame, "power")), ""
        if not cells or not self.dig:
            return None
        for c in cells[:-1]:
            best = max(((_score(c, t), d) for d, t in self.dig.items()), default=(0, None))
            if best[0] < MIN_SCORE:
                return None
            s += str(best[1])
        # the last cell must be the % sign: it matches no digit
        if max(_score(cells[-1], t) for t in self.dig.values()) >= MIN_SCORE or not s or len(s) > 3:
            return None
        v = int(s)
        return float(v) if v <= 100 else None

    def read_usage(self, frame):
        r = self._roi(frame, "usage").astype(int)
        if r.shape[1] < USAGE_SEGS * 2 or r.shape[0] < 4:
            return None
        w = r.shape[1] // USAGE_SEGS
        lit = [(((r[:, i * w:(i + 1) * w, 1] > BAR_G) & (r[:, i * w:(i + 1) * w, 2] < BAR_B)).mean() > BAR_FRAC) for i in range(USAGE_SEGS)]
        n = 0
        while n < USAGE_SEGS and lit[n]:
            n += 1
        return n if n >= 1 and not any(lit[n:]) else None

    def read(self, frame, t):
        return Readings(t, hour=self.read_hour(frame), power_pct=self.read_power(frame), usage=self.read_usage(frame))


def hour_period_s(times, hours):
    """Median seconds between consecutive hour changes, from per-frame (t, hour) with None gaps."""
    last, change = None, []
    for t, h in zip(times, hours):
        if h is None:
            continue
        if last is not None and h != last[1]:
            change.append(t)
        last = (t, h)
    return float(np.median(np.diff(change))) if len(change) >= 2 else None
