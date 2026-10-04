"""Phase 14: hallway occupancy by pixel difference against a frozen empty-hallway reference.
Only meaningful with the hall light on, so light_on not True -> None."""
import json
from pathlib import Path

import numpy as np

from geometry import roi
from readers.digits import _labeled
from state import Readings

MIN_REFS = 20
LIT_MIN = 3.0  # mean gray of a lit hallway ROI is ~11; ~0.3 means the light is not actually on yet/anymore


def _gray(frame, geom, side, buttons):
    return roi(frame, geom, "hall_" + side, buttons).astype(np.float32).mean(axis=2)


def build_references(session_dirs, buttons, out="data/templates/hall_ref.npz", keep=lambda sess, i: True):
    """Mean grayscale ROI over frames labeled hall=0 with the light on, per side. Frozen at runtime."""
    acc = {"L": [], "R": []}
    for d in sorted(map(Path, session_dirs)):
        for frame, geom, row in _labeled([d], lambda i, d=d: keep(d.name, i)):
            for s in "LR":
                if row.get("hall_" + s) == "0" and row.get("light_" + s) == "1":
                    g = _gray(frame, geom, s, buttons)
                    if g.mean() >= LIT_MIN:
                        acc[s].append(g)
    for s, v in acc.items():
        if len(v) < MIN_REFS:
            raise ValueError(f"hall_{s}: {len(v)} empty reference frames, need >= {MIN_REFS}")
    refs = {s: np.mean(v, axis=0) for s, v in acc.items()}
    # noise: spread of the per-frame MSE of the empty frames themselves (leave-in; refs are means of many frames)
    stats = {}
    for s in "LR":
        e = [float(((g - refs[s]) ** 2).mean()) for g in acc[s]]
        stats[s] = (float(np.mean(e)), float(np.std(e)) or 1.0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, refL=refs["L"], refR=refs["R"], noiseL=stats["L"], noiseR=stats["R"])
    return {s: len(acc[s]) for s in "LR"}


class HallwayDetector:
    def __init__(self, geom, buttons, refs_path="data/templates/hall_ref.npz", thresh=None):
        z = np.load(refs_path)
        self.refs = {s: z["ref" + s] for s in "LR"}
        self.noise = {s: tuple(z["noise" + s]) for s in "LR"}
        self.geom, self.buttons = geom, buttons
        self.thresh = thresh or load_thresh() or {"L": 150.0, "R": 150.0}  # MSE units (noise sigma is 1 for this static scene)

    def sigmas(self, frame, side):
        g = _gray(frame, self.geom, side, self.buttons)
        ref = self.refs[side]
        if g.shape != ref.shape or g.mean() < LIT_MIN:  # unlit ROI says nothing (light button state can lead the picture)
            return None
        mu, sd = self.noise[side]
        return (float(((g - ref) ** 2).mean()) - mu) / sd

    def score(self, frame, side, light_on):
        """-> occupancy in [0,1] with 0.5 exactly at the fitted threshold, or None."""
        if light_on is not True:
            return None
        z = self.sigmas(frame, side)
        return None if z is None else float(1 / (1 + np.exp(-(z - self.thresh[side]) / max(1.0, self.thresh[side] / 4))))

    def read(self, frame, t, light_on=None):
        # the supervisor only trusts hall scores while it knows the light is on; ButtonReader supplies that upstream
        return Readings(t, hall={s: self.score(frame, s, (light_on or {}).get(s)) for s in "LR"})


def fit_threshold(scores, labels):
    """-> (threshold at max Youden J, AUC, J). labels 1 = occupied."""
    from sklearn.metrics import roc_auc_score, roc_curve
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(roc_auc_score(labels, scores)), float(j[k])


def dwell_stats(times, labels):
    """Seconds of each contiguous run of occupied=1 in a (time-sorted) label series; None gaps break runs."""
    runs, start, last = [], None, None
    for t, v in zip(times, labels):
        if v == 1 and start is None:
            start = t
        elif v != 1 and start is not None:
            runs.append(last - start)
            start = None
        last = t
    if start is not None:
        runs.append(last - start)
    return runs


def save_fit(side, thresh, auc, noise, path="config_fit.json"):
    p = Path(path)
    d = json.loads(p.read_text()) if p.exists() else {}
    d.setdefault("hall_thresh", {})[side] = thresh
    d.setdefault("hall_auc", {})[side] = auc
    d.setdefault("hall_noise_sigma", {})[side] = noise
    p.write_text(json.dumps(d, indent=1))


def load_thresh(path="config_fit.json"):
    p = Path(path)
    t = json.loads(p.read_text()).get("hall_thresh") if p.exists() else None
    return t or None


def fit_all(session_dirs, buttons, refs_out="data/templates/hall_ref.npz", fit_path="config_fit.json"):
    """Refs from every lit empty frame, thresholds at the midpoint between the classes' MSE.
    A side with no occupied examples (R, so far) borrows the other side's threshold; its AUC is not measured."""
    n = build_references(session_dirs, buttons, refs_out)
    z = np.load(refs_out)
    scores = {"L": ([], []), "R": ([], [])}
    for d in sorted(map(Path, session_dirs)):
        for frame, geom, row in _labeled([d]):
            for s in "LR":
                if row.get("hall_" + s) in ("0", "1") and row.get("light_" + s) == "1":
                    g = _gray(frame, geom, s, buttons)
                    if g.mean() >= LIT_MIN:
                        scores[s][0].append(float(((g - z["ref" + s]) ** 2).mean()))
                        scores[s][1].append(int(row["hall_" + s]))
    out = {}
    for s, (sc, y) in scores.items():
        sc, y = np.array(sc), np.array(y)
        if y.sum() and (y == 0).sum():
            _, auc, _ = fit_threshold(sc, y)
            out[s] = ((sc[y == 0].max() + sc[y == 1].min()) / 2, auc)
    for s in "LR":
        thr, auc = out.get(s, (out[next(iter(out))][0], None))
        save_fit(s, float(thr), auc, float(z["noise" + s][1]), fit_path)
    return n, out


if __name__ == "__main__":
    import config
    from geometry import load_buttons
    print(fit_all(["data/corpus/n1_a", "data/corpus/n2_a"], load_buttons(config.CALIBRATION_PATH)))
