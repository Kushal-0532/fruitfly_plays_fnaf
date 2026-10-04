import json
import random

import numpy as np
import pytest

from config import REF_H, REF_W
from geometry import (REQUIRED_BUTTONS, Geometry, Uncalibrated, Window, capture_to_ref, center,
                      in_window, load_buttons, ref_to_window, roi, window_to_capture)


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (960, 540)])
@pytest.mark.parametrize("scale", [1, 2])
def test_roundtrip_matrix(size, scale):
    g = Geometry(Window(1, 0, 0, *size), scale)
    rnd = random.Random(0)
    for _ in range(100):
        p = (rnd.uniform(0, REF_W - 1), rnd.uniform(0, REF_H - 1))
        q = capture_to_ref(g, *window_to_capture(g, *ref_to_window(g, *p)))
        assert abs(q[0] - p[0]) <= 2.001 and abs(q[1] - p[1]) <= 2.001


def test_out_of_window_detected():
    g = Geometry(Window(1, 0, 0, 1920, 1080))
    for p in [(-50, 10), (10, -50), (REF_W + 50, 10), (10, REF_H + 50)]:
        assert not in_window(g, *ref_to_window(g, *p))
    assert in_window(g, *ref_to_window(g, 640, 360))


def _write(tmp_path, d):
    f = tmp_path / "c.json"
    f.write_text(json.dumps(d))
    return f


def _good():
    return {"ref": [1280, 720], "buttons": {n: [10, 10, 20, 20] for n in REQUIRED_BUTTONS}}


def test_uncalibrated_template_refused():
    with pytest.raises(Uncalibrated):
        load_buttons("calibration.example.json")


def test_calibrated_loads(tmp_path):
    b = load_buttons(_write(tmp_path, _good()))
    assert b["door_L"] == (10, 10, 20, 20) and center(b["door_L"]) == (20, 20)


def test_missing_required_button(tmp_path):
    d = _good()
    del d["buttons"]["door_L"]
    with pytest.raises(Uncalibrated):
        load_buttons(_write(tmp_path, d))


def test_roi_slice():
    g = Geometry(Window(1, 0, 0, 1280, 720), 2)
    frame = np.arange(360 * 640).reshape(360, 640)
    out = roi(frame, g, "a", {"a": (100, 50, 40, 20)})
    assert np.array_equal(out, frame[25:35, 50:70])


def test_roi_outside_raises():
    g = Geometry(Window(1, 0, 0, 1280, 720), 2)
    with pytest.raises(ValueError):
        roi(np.zeros((360, 640)), g, "a", {"a": (2000, 2000, 40, 20)})
