import csv
import json
import subprocess
import sys

import numpy as np
import pytest

import label
from calibrate import build_calibration
from geometry import REQUIRED_BUTTONS, Geometry, Window
from label import COLUMNS, make_sheets, make_template, ready_failures, validate
from record import load_session, record_session
from tests.fakes import FakeClock


class SeqCapture:
    def __init__(self):
        self.rnd, self.frames = np.random.RandomState(0), []

    def grab(self):
        f = self.rnd.randint(0, 255, (36, 64, 3), dtype=np.uint8)
        self.frames.append(f)
        return f


def make_session(d, n=30):
    cap, c = SeqCapture(), FakeClock()
    record_session(cap, d, n / 10 - 0.01, clock=c.now, sleep=c.sleep)
    return cap


def test_record_roundtrip(tmp_path):
    cap = make_session(tmp_path / "s", 30)
    meta, it = load_session(tmp_path / "s")
    got = list(it)
    assert len(got) == len(cap.frames) == len(meta["times"]) == 30
    assert all(np.array_equal(f, g[1]) for f, g in zip(cap.frames, got))
    assert [g[0] for g in got] == sorted(g[0] for g in got)


def test_calibration_conversion():
    g = Geometry(Window(1, 0, 0, 1920, 1080), 2)
    samples = {n: [(960, 540)] for n in REQUIRED_BUTTONS}
    samples["hall_L"] = [(300, 150), (600, 450)]
    cal = build_calibration(samples, g)
    assert cal["uncalibrated"] is False
    assert cal["buttons"]["door_L"] == [620, 340, 40, 40]   # centre (640,360) ref
    assert cal["buttons"]["hall_L"] == [200, 100, 200, 200]


def test_calibration_outside_refused():
    g = Geometry(Window(1, 0, 0, 1920, 1080), 2)
    with pytest.raises(ValueError):
        build_calibration({"door_L": [(2000, 10)]}, g)
    with pytest.raises(ValueError):
        build_calibration({"door_L": [(-1, 10)]}, g)


def test_label_template_sheet(tmp_path):
    make_session(tmp_path / "s", 100)
    p = make_template(tmp_path / "s", every=10)
    rows = list(csv.reader(open(p)))
    assert rows[0] == COLUMNS and len(rows) - 1 == 10
    sheets = make_sheets(tmp_path / "s", cols=4, rows=2, tile_w=64)
    assert len(sheets) == 2
    from PIL import Image
    assert Image.open(sheets[0]).size == (4 * 64, 2 * 36) and Image.open(sheets[1]).size == (4 * 64, 36)


def write_labels(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            d = dict.fromkeys(COLUMNS, "")
            d.update(r)
            w.writerow([d[c] for c in COLUMNS])


def test_label_check_validation(tmp_path):
    p = tmp_path / "labels.csv"
    write_labels(p, [{"frame": 0, "hour": 2, "power_pct": 80, "door_L": 1, "cam": "1A", "hall_L": 0}])
    assert validate(p)[0] == []
    for bad in ({"door_L": 2}, {"hall_R": 2}, {"cam": "9Z"}, {"hour": 7}, {"foxy_stage": 9}):
        write_labels(p, [{"frame": 0, **bad}])
        assert validate(p)[0], bad
    p.write_text("frame,hour\n0,1\n")
    assert any("missing column" in e for e in validate(p)[0])


def build_corpus(root, cal):
    """A corpus that satisfies every READY minimum."""
    for i in range(3):
        d = root / f"n1_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"times": []}))
        rows = []
        for k in range(60):
            rows.append({"frame": k, "hour": 1, "power_pct": 90, "monitor_up": k % 2, "hall_L": k % 2,
                         "hall_R": (k // 2) % 2, "foxy_stage": 0})
        write_labels(d / "labels.csv", rows)
    (root / "foxy_stages.txt").write_text("active_night1: yes\nside: L\nstage 0: a\nstage 1: b\n")
    cal.write_text(json.dumps({"ref": [1280, 720], "buttons": {n: [1, 1, 2, 2] for n in REQUIRED_BUTTONS}}))


def test_ready_minimums(tmp_path):
    root, cal = tmp_path / "corpus", tmp_path / "cal.json"
    build_corpus(root, cal)
    assert ready_failures(root, cal) == []
    breakers = {
        "sessions": lambda: __import__("shutil").rmtree(root / "n1_2"),
        "hall": lambda: write_labels(root / "n1_0" / "labels.csv", []) or write_labels(root / "n1_1" / "labels.csv", []),
        "monitor": lambda: [write_labels(root / f"n1_{i}" / "labels.csv",
                                         [{"frame": k, "hour": 1, "power_pct": 9, "monitor_up": 1, "hall_L": k % 2,
                                           "hall_R": k % 2} for k in range(60)]) for i in range(3)],
        "hourpower": lambda: [write_labels(root / f"n1_{i}" / "labels.csv",
                                           [{"frame": k, "monitor_up": k % 2, "hall_L": k % 2, "hall_R": k % 2}
                                            for k in range(60)]) for i in range(3)],
        "calibration": lambda: cal.write_text(json.dumps({"uncalibrated": True, "buttons": {}})),
        "foxy": lambda: (root / "foxy_stages.txt").unlink(),
    }
    for name, brk in breakers.items():
        build_corpus(root, cal)
        brk()
        assert ready_failures(root, cal), name


def test_ready_cli_writes_file(tmp_path, monkeypatch):
    import config
    root, cal = tmp_path / "data" / "corpus", tmp_path / "calibration.json"
    build_corpus(root, cal)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        label.main(["check", "n1_0", "--ready"])
    assert e.value.code == 0 and (root / "READY").exists()
    (root / "READY").unlink()
    (root / "foxy_stages.txt").unlink()
    with pytest.raises(SystemExit) as e:
        label.main(["check", "n1_0", "--ready"])
    assert e.value.code == 1 and not (root / "READY").exists()


def test_fast_import():
    code = ("import time,sys;t=time.time();import record,calibrate,label;"
            "assert time.time()-t<1,time.time()-t;"
            "assert not {'torch','flyvis'}&set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)
