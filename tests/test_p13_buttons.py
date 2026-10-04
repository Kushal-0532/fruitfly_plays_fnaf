from pathlib import Path

import numpy as np
import pytest

from geometry import Geometry, Window, roi
from readers.buttons import ButtonReader

GEOM = Geometry(Window(1, 0, 0, 1280, 720), 2)
B = {"door_L": (45, 336, 40, 40), "door_R": (1196, 344, 40, 40), "light_L": (42, 437, 40, 40), "light_R": (1193, 442, 40, 40),
     "cam_map": (831, 340, 394, 328), "cam_1A": (966, 347, 40, 40), "cam_1B": (940, 400, 40, 40), "cam_1C": (912, 478, 40, 40)}
from readers.buttons import CAMS
for _i, _c in enumerate(c for c in CAMS if c not in ("1A", "1B", "1C")):
    B["cam_" + _c] = (840 + 45 * _i, 600, 40, 40)
RECTS = {n: tuple(v // 2 for v in B[n]) for n in B}  # capture px (scale 2, window == ref size)


def blank():
    return np.zeros((360, 640, 3), np.uint8)


def paint(f, name, rgb):
    x, y, w, h = RECTS[name]
    f[y:y + h, x:x + w] = rgb


def test_door_and_light_colours():
    f = blank()
    paint(f, "door_L", (30, 230, 30))   # green = closed
    paint(f, "light_L", (60, 80, 230))  # blue = on
    paint(f, "door_R", (230, 20, 20))   # red = open, light off
    r = ButtonReader(GEOM, B).read(f, 0)
    assert r.door_closed == {"L": True, "R": False} and r.light_on == {"L": True, "R": False} and r.monitor_up is False


def test_none_on_noise():
    rng = np.random.default_rng(1)
    for f in (blank(), np.full((360, 640, 3), 128, np.uint8)):
        r = ButtonReader(GEOM, B).read(f, 0)
        assert r.door_closed == {"L": None, "R": None} and r.light_on == {"L": None, "R": None}
    r = ButtonReader(GEOM, B).read(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), 0)
    assert r.monitor_up is True and r.door_closed is None  # white-noise ROIs look like the camera map; never claims door state


def test_monitor_up_and_cam():
    f = blank()
    f[...] = 20
    x, y, w, h = RECTS["cam_map"]
    f[y:y + h:8, x:x + w] = 255  # white map lines
    paint(f, "cam_1B", (190, 210, 20))
    r = ButtonReader(GEOM, B).read(f, 0)
    assert r.monitor_up is True and r.cam == "1B" and r.door_closed is None
    paint(f, "cam_1A", (190, 210, 20))  # two highlights at once -> ambiguous
    assert ButtonReader(GEOM, B).read(f, 0).cam is None


@pytest.mark.corpus
def test_corpus_accuracy():
    import config
    from geometry import load_buttons
    from readers.digits import _labeled
    b = load_buttons(config.CALIBRATION_PATH)
    ok = bad = n = 0
    for sess in ("n1_a", "n2_a"):
        r = None
        for frame, geom, row in _labeled([Path("data/corpus") / sess]):
            r = r or ButtonReader(geom, b)
            if row["monitor_up"].strip():
                n += 1
                got = r.monitor_up(frame)
                ok += got == (row["monitor_up"] == "1")
                bad += got != (row["monitor_up"] == "1")
    assert ok / n >= 0.99 and bad / n <= 0.02
