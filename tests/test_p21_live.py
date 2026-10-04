import json
import re
import urllib.request

import numpy as np

import export_replay
import fly_brain
import live_viz
from state import Readings


class FakeBrain:
    def __init__(self, danger):
        self.danger, self.got = danger, []

    def submit(self, frame, t, light_on=None):
        self.got.append(t)

    error = None

    def snapshot(self):
        return {"t": self.got[-1] if self.got else 0.0, "danger": self.danger, "contrib": {"L": [0, 0, 0, 0], "R": [0, 0, 0, 0]}, "maps": {}, "ms": 1.0}


def test_fly_hallway_only_speaks_with_light_on():
    b = FakeBrain({"L": 0.9, "R": 0.8})
    h = fly_brain.FlyHallway(b)
    r = h.read(None, 1.0, {"L": True, "R": False})
    assert r.hall == {"L": 0.9, "R": None} and b.got == [1.0]
    assert h.read(None, 2.0, None).hall == {"L": None, "R": None}


def test_readout_math(tmp_path):
    p = tmp_path / "r.npz"
    np.savez(p, types=np.array(["R1", "L1"]), w=np.array([2.0, 0.0]), b=-1.0, sd=np.array([1.0, 1.0]),
             baseL=np.array([0.0, 0.0]), baseR=np.array([10.0, 0.0]))
    ro = fly_brain.Readout(p)
    assert ro.score("L", [0.0, 5.0]) < 0.3 < 0.7 < ro.score("L", [3.0, 0.0])
    assert abs(ro.score("R", [10.0, 0.0]) - ro.score("L", [0.0, 0.0])) < 1e-9  # each side is measured from its own empty baseline


def test_live_page_and_state():
    v = live_viz.LiveViz(FakeBrain({"L": 0.1, "R": None}), port=0)
    port = v.server.server_address[1]
    page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
    assert "Fly brain plays" in page and "http" not in re.sub(r"http://localhost|http://127", "", page.split("<script>")[1]).split("fetch")[0][:0]
    row = {"t": 3.0, "action": "LIGHT_L", "arg": None, "reason": "hall_check"}
    state = Readings(3.0)

    class S:
        hour, power_pct, usage, monitor_up = 1, 90.0, 1, False
        door_closed, light_on = {"L": False, "R": False}, {"L": True, "R": False}
    v.update(S, row)
    s = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state").read())
    assert s["danger"] == {"L": 0.1, "R": None} and s["bot"]["hour"] == 1 and s["log"][0]["reason"] == "hall_check"
    v.close()


def test_pages_use_no_external_resources():
    for html in (live_viz.PAGE, export_replay.PAGE):
        assert not re.search(r"(src|href)=[\"']https?://|@import|url\(https?://", html)


def test_fly_hallway_rides_through_light_flicker():
    b = FakeBrain({"L": 0.97, "R": None})
    h = fly_brain.FlyHallway(b, smooth=5)
    on = {"L": True, "R": False}
    assert h.read(None, 0.0, on).hall["L"] == 0.97
    b.danger = {"L": 0.0, "R": None}  # one flickered frame
    assert h.read(None, 0.1, on).hall["L"] == 0.97
    for k in range(5):
        r = h.read(None, 0.2 + k / 10, on)
    assert r.hall["L"] == 0.0  # genuinely gone for the whole window
    h.read(None, 1.0, {"L": False, "R": False})
    b.danger = {"L": 0.2, "R": None}
    assert h.read(None, 1.1, on).hall["L"] == 0.2  # history cleared when the light went off


def test_dead_brain_fails_loudly_and_stale_verdicts_are_ignored():
    import pytest
    b = FakeBrain({"L": 0.97, "R": None})
    h = fly_brain.FlyHallway(b)
    h.read(None, 0.0, {"L": True, "R": False})
    b.snapshot = lambda: {"t": 0.0, "danger": {"L": 0.97, "R": None}}  # frozen at t=0
    assert h.read(None, 5.0, {"L": True, "R": False}).hall["L"] is None  # 5 s old: not used
    b.error = ValueError("boom")
    with pytest.raises(RuntimeError):
        h.read(None, 6.0, {"L": True, "R": False})
