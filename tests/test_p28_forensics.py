import json

import numpy as np
from PIL import Image

import config
from scripts import auto_night
from scripts.forensics import COLS, TW, make_sheet, pick_frames


def row(t, trusted=True, reason="idle"):
    return {"t": t, "hour": 1, "power_pct": 80.0, "usage": 1, "monitor_up": False, "door_closed": {"L": False, "R": False},
            "light_on": {"L": None, "R": None}, "cam": None, "hall": {"L": 0.1, "R": None}, "trusted": trusted,
            "action": "NONE", "reason": reason, "guard": ""}


def fake_run(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "frames").mkdir(parents=True)
    for i in range(12):
        Image.new("RGB", (64, 36), (i * 20, 0, 0)).save(corpus / "frames" / f"{i:06d}.png")
    (corpus / "meta.json").write_text(json.dumps({"times": [i / 10 for i in range(12)]}))
    log = tmp_path / "run_20260101-000000.jsonl"
    rows = [row(i / 10, trusted=i < 8, reason="jam_L" if i == 3 else "idle") for i in range(12)]
    log.write_text("\n".join(json.dumps(x) for x in [{"header": {}}, *rows]) + "\n")
    return log, corpus


def test_pick_frames_dedups_and_orders():
    assert pick_frames([0.0, 0.5, 1.0], 1.0, 1.0, 10) == [0, 1, 2]


def test_sheet_and_summary(tmp_path):
    log, corpus = fake_run(tmp_path)
    p = make_sheet(log, corpus, tmp_path / "out", seconds=0.7, fps=10)
    im = Image.open(p)
    n = 8 + 4  # frames 0..7 up to t_end=0.7, then 4 after-death frames
    assert im.width == COLS * TW and im.height % -(-n // COLS) == 0
    s = json.load(open(tmp_path / "out" / "summary.json"))
    assert {"last_trusted", "last_actions", "monitor_raised_last_10s", "jam_events", "door_audit"} <= set(s)
    assert s["jam_events"] == [[0.3, "jam_L"]]


def test_auto_night_writes_sheet(tmp_path, monkeypatch):
    log, corpus = fake_run(tmp_path)
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "CORPUS_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run_20260101-000000.jsonl").write_text(log.read_text())
    (tmp_path / "x_1").symlink_to(corpus)
    monkeypatch.setattr(auto_night.np, "load", lambda p: {})
    monkeypatch.setattr(auto_night, "wait_menu", lambda *a: True)
    monkeypatch.setattr(auto_night, "click_continue", lambda: None)
    monkeypatch.setattr(auto_night, "raise_game", lambda: None)
    monkeypatch.setattr(auto_night.time, "time", lambda: 1.0)
    monkeypatch.setattr(auto_night.time, "sleep", lambda s: None)

    class P:
        returncode = 0
        def poll(self): return 0
        def communicate(self): return ("RunSummary(status='x')", "")
    monkeypatch.setattr(auto_night.subprocess, "Popen", lambda *a, **k: P())
    auto_night.main(["--night", "2", "--rounds", "1", "--", "--record", "x"])
    rec = json.loads((tmp_path / "logs" / "auto" / "rounds.jsonl").read_text().splitlines()[-1])
    assert rec["sheet"].endswith("sheet.png") and "sheet_error" not in rec
