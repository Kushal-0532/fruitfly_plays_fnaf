import types

import numpy as np

import config
import fly_brain
from fly_brain import FlyHallway, cove_box, stimulus
from state import Readings, StateTracker


def frame_and_ref():
    f = np.random.RandomState(0).randint(0, 255, (100, 200, 3)).astype(np.uint8)
    y0, y1, x0, x1 = cove_box(f)
    return f, fly_brain.cove_gray(f), (y0, y1, x0, x1)


def test_stimulus_diff_only_on_1c_cam_region():
    f, ref, (y0, y1, x0, x1) = frame_and_ref()
    out = stimulus(f, None, None, None, None, "1C", ref)
    assert out[y0:y1, x0:x1].max() == 0 and (out[:, x1 + 2:] == f[:, x1 + 2:]).all()
    g = f.copy()
    g[y0 + 5:y0 + 15, x0 + 5:x0 + 15] = np.clip(g[y0 + 5:y0 + 15, x0 + 5:x0 + 15].astype(int) + 60, 0, 255)
    d = stimulus(g, None, None, None, None, "1C", ref)
    assert d[y0 + 5:y0 + 15, x0 + 5:x0 + 15].min() > 0 and d[y0 + 20:y0 + 30, x0 + 20:x0 + 30].max() == 0
    assert (stimulus(f, None, None, None, None, "2A", ref) == f).all()


def test_fit_cove_separable(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    root = tmp_path / "corpus" / "s"
    root.mkdir(parents=True)
    (tmp_path / "data" / "features").mkdir(parents=True)
    rows, feats, frames = ["frame,stage,look_id"], [], []
    for i in range(120):
        stage = 1 if i < 60 else 3
        rows.append(f"{i},{stage},{i // 10}")
        feats.append(rng.normal(0, 1, 6) + (3 if stage == 3 else 0)); frames.append(i)
    (root / "labels_cove.csv").write_text("\n".join(rows))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CORPUS_DIR", str(tmp_path / "corpus"))
    np.savez("data/features/fly_cove_diff_s.npz", frames=np.array(frames), look=np.arange(120) // 10, feats=np.array(feats))
    res = fly_brain.fit_cove(["s"], out=str(tmp_path / "c.npz"))
    assert res["heldout_look_auc"] >= 0.99 and res["stage1_false_alarm"] <= 0.02
    assert {"use", "w", "b", "mean", "sd", "threshold", "mode"} <= set(np.load(tmp_path / "c.npz").files)


class FakeBrain:
    error = None
    def __init__(self, snap): self.snap = snap
    def submit(self, *a): pass
    def snapshot(self): return self.snap


def test_cam_read_cove_gating():
    snap = {"t": 10.0, "cam_feat": [0] * 6, "cam": "1C", "cove": 0.9}
    assert FlyHallway(FakeBrain(snap)).cam_read(None, 10.2, "1C").cove == 0.9
    assert FlyHallway(FakeBrain(snap)).cam_read(None, 10.2, "2A").cove is None      # wrong cam
    assert FlyHallway(FakeBrain({**snap, "cam": "2A"})).cam_read(None, 10.2, "1C").cove is None  # snapshot made for another cam
    assert FlyHallway(FakeBrain(snap)).cam_read(None, 11.0, "1C").cove is None      # stale


def test_state_holds_cove_then_drops_it():
    tr = StateTracker(types.SimpleNamespace(**vars(config)))
    tr.update(Readings(0.0, cove=0.8))
    assert tr.update(Readings(10.0)).cove == 0.8
    assert tr.update(Readings(0.0 + config.STALE_S["cove"] + 1)).cove is None
