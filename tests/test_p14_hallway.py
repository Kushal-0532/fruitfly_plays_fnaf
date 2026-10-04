import numpy as np
import pytest

from geometry import Geometry, Window
from readers import hallway as H

GEOM = Geometry(Window(1, 0, 0, 1280, 720), 2)
B = {"hall_L": (80, 100, 190, 400), "hall_R": (990, 130, 150, 400)}


def test_fit_threshold_synthetic():
    thr, auc, j = H.fit_threshold([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert auc == 1.0 and j == 1.0 and 3 < thr <= 10
    rng = np.random.default_rng(0)
    _, auc, _ = H.fit_threshold(rng.normal(size=400), rng.integers(0, 2, 400))
    assert abs(auc - 0.5) < 0.1


def _det(tmp_path, lit_level=11.0):
    p = tmp_path / "r.npz"
    ref = np.full((200, 95), lit_level, np.float32)
    np.savez(p, refL=ref, refR=ref, noiseL=(0.0, 1.0), noiseR=(0.0, 1.0))
    return H.HallwayDetector(GEOM, B, p, {"L": 150.0, "R": 150.0})


def frame_with(level, block=None):
    f = np.full((360, 640, 3), level, np.uint8)
    if block:
        f[60:260, 40:135][::2] = block
    return f


def test_light_off_none(tmp_path):
    d = _det(tmp_path)
    f = frame_with(11)
    assert d.score(f, "L", False) is None and d.score(f, "L", None) is None
    assert d.score(frame_with(0), "L", True) is None  # light says on but the picture is dark


def test_occupancy_score(tmp_path):
    d = _det(tmp_path)
    empty, occupied = d.score(frame_with(11), "L", True), d.score(frame_with(11, block=200), "L", True)
    assert empty < 0.1 and occupied > 0.9


def test_reference_minimum(tmp_path):
    d = tmp_path / "s"
    (d / "frames").mkdir(parents=True)
    from PIL import Image
    Image.fromarray(frame_with(11)).save(d / "frames" / "000000.png")
    (d / "meta.json").write_text('{"window": {"wid": 1, "x": 0, "y": 0, "w": 1280, "h": 720}, "capture_scale": 2}')
    (d / "labels.csv").write_text("frame,hall_L,hall_R,light_L,light_R\n0,0,0,1,1\n")
    with pytest.raises(ValueError):
        H.build_references([d], B, out=tmp_path / "x.npz")


def test_dwell_stats():
    assert H.dwell_stats([0, 1, 2, 3, 4, 5], [0, 1, 1, 0, 1, 1]) == [1, 1]
