import json
from pathlib import Path

import numpy as np
import pytest

from geometry import Geometry, Window
from readers import digits as D

GEOM = Geometry(Window(1, 0, 0, 1280, 720), 2)
BUTTONS = {"clock": (1000, 50, 100, 30), "power": (180, 620, 60, 30), "usage": (120, 655, 80, 30)}


def glyph(rows=10, cols=6):
    g = np.zeros((rows, cols), bool)
    g[0, :], g[:, 0] = True, True  # an "L"-ish shape
    return g


def test_segmentation_synthetic():
    m = np.zeros((16, 40), np.uint8)
    m[3:13, 2:8][glyph()] = 255
    m[3:13, 12:18][glyph()] = 255
    cells = D.split_digits(m)
    assert len(cells) == 2 and cells[0].shape == D.CELL and (cells[0] == cells[1]).all()


def make_reader(tmp_path):
    p = tmp_path / "d.npz"
    c = np.zeros(D.CELL, bool)
    c[:10, :6] = glyph()
    np.savez(p, d1=c, h1=np.ones((14, D.HOUR_COLS), bool))
    return D.DigitReader(GEOM, BUTTONS, p), c


def test_none_on_noise(tmp_path):
    r, _ = make_reader(tmp_path)
    rng = np.random.default_rng(0)
    for f in (np.zeros((360, 640, 3), np.uint8), np.full((360, 640, 3), 128, np.uint8),
              rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)):
        assert r.read_hour(f) is None and r.read_power(f) is None


def test_power_reads_digit_then_percent(tmp_path):
    r, c = make_reader(tmp_path)
    f = np.zeros((360, 640, 3), np.uint8)
    x0, y0 = 92, 312  # inside the power ROI (capture px)
    f[y0:y0 + 10, x0:x0 + 6][glyph()] = 255
    f[y0:y0 + 9, x0 + 9:x0 + 18] = 255  # solid block: matches no digit, plays the % sign
    assert r.read_power(f) == 1.0


def test_usage_counts_bars(tmp_path):
    r, _ = make_reader(tmp_path)
    f = np.zeros((360, 640, 3), np.uint8)
    ys, xs = slice(327, 343), slice(60, 101)  # usage ROI in capture px
    f[ys, xs][:, :20] = (30, 230, 30)  # first two of four segments lit
    assert r.read_usage(f) == 2


def test_hour_period():
    ts = list(range(0, 400, 5))
    hs = [t // 89 for t in ts]
    assert abs(D.hour_period_s(ts, hs) - 89) <= 5


CORPUS = Path("data/corpus")


@pytest.mark.corpus
def test_corpus_accuracy():
    """Templates from even 10-frame blocks, scored on odd blocks (only 2 sessions exist, so no true held-out session)."""
    from geometry import load_buttons
    import config
    b = load_buttons(config.CALIBRATION_PATH)
    tmp = Path("data/templates/_test_digits.npz")
    D.build_templates([CORPUS / "n1_a", CORPUS / "n2_a"], b, out=tmp, keep=lambda i: (i // 10) % 2 == 0)
    ok, wrong, n, r = 0, 0, 0, None
    for frame, geom, row in D._labeled([CORPUS / "n1_a"], lambda i: (i // 10) % 2 == 1):
        r = r or D.DigitReader(geom, b, tmp)
        for fn, lab in ((r.read_hour, row["hour"]), (r.read_power, row["power_pct"])):
            if lab.strip():
                v, n = fn(frame), n + 1
                ok += v is not None and v == float(lab)
                wrong += v is not None and v != float(lab)
    tmp.unlink()
    assert ok / n >= 0.98 and wrong / n <= 0.02
