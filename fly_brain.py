"""The fly as the bot's perception of danger.

A flyvis network watches every game frame (background thread). For each hallway we read the activity of a few of its own
neurons (L1, L2, Mi1 and the photoreceptors R1) over the patch of retina that looks at that hallway, and a small logistic
readout on those four numbers is the "danger" score the supervisor acts on. Nothing here looks at pixels directly.

  python fly_brain.py            # (re)fit the readout from the labeled corpus -> data/templates/fly_readout.npz
"""
import base64
import io
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import config
from geometry import ref_to_window, roi_box, window_to_capture
from state import Readings

T4, T5 = ("T4a", "T4b", "T4c", "T4d"), ("T5a", "T5b", "T5c", "T5d")
READOUT_TYPES = ("R1", "L1", "L2", "Mi1")   # the neurons whose regional activity is the danger signal
VIEW_TYPES = ("R1", "L1", "L2", "Mi1", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
CAM_VIEW = (0.02, 0.62, 0.10, 0.88)  # x0, x1, y0, y1 as fractions of the frame: the camera picture, left of the map
READOUT_PATH = "data/templates/fly_readout.npz"
REFS_PATH = "data/templates/hall_ref.npz"
COVE_REF_PATH = "data/templates/cove_ref.npz"
COVE_PATH = "data/templates/fly_cove.npz"
COVE_GONE_PATH = "data/templates/fly_cove_gone.npz"
GONE_GRID = (5, 4)  # the gone readout pools the fly's cam-1C activity per type over this grid (x, y): WHERE the activity is tells
                    # "Foxy out on the stage" (3) from "cove empty" (4); the 6 view-wide means could not (held-out hit 4-16%)
CAM4B_REF_PATH = "data/templates/cam4b_ref.npz"
CAM4B_PATH = "data/templates/fly_cam4b.npz"
CAM_PATHS = {"1C": (COVE_REF_PATH, COVE_PATH), "4B": (CAM4B_REF_PATH, CAM4B_PATH)}  # cam -> (empty reference, fitted readout)
DIFF_GAIN = 4.0   # |frame - empty hallway| is amplified into the fly's input
LIT_MIN = 3.0     # mean gray below this = the hallway light is not really on (flicker / switching)
JPEG_W = 320


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def hex_frame_coords(perception, h=360, w=640):
    """Where each of the 721 hexals looks, as fractions of the frame (measured by feeding gradients through the eye)."""
    xg = perception.to_hex(np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)).numpy().reshape(-1)
    yg = perception.to_hex(np.tile(np.linspace(0, 255, h)[:, None], (1, w)).astype(np.uint8)).numpy().reshape(-1)
    return xg, yg


def region_masks(frame_shape, geom, buttons, xg, yg):
    """Boolean mask over the 721 hexals whose field of view falls inside each hallway ROI."""
    fh, fw = frame_shape[:2]
    out = {}
    for s in "LR":
        rx, ry, rw, rh = buttons["hall_" + s]
        x0, y0 = window_to_capture(geom, *ref_to_window(geom, rx, ry))
        x1, y1 = window_to_capture(geom, *ref_to_window(geom, rx + rw, ry + rh))
        out[s] = (xg >= x0 / fw) & (xg <= x1 / fw) & (yg >= y0 / fh) & (yg <= y1 / fh)
    return out


def load_refs(path=REFS_PATH):
    """Empty-hallway reference pictures per side (the same ones the pixel detector uses), or None."""
    if not Path(path).exists():
        return None
    z = np.load(path)
    return {"L": z["refL"], "R": z["refR"]}


def cove_box(frame):
    h, w = frame.shape[:2]
    return int(CAM_VIEW[2] * h), int(CAM_VIEW[3] * h), int(CAM_VIEW[0] * w), int(CAM_VIEW[1] * w)


def cove_gray(frame):
    y0, y1, x0, x1 = cove_box(frame)
    return frame[y0:y1, x0:x1].astype(np.float32).mean(axis=2)


def load_cove_ref(path=COVE_REF_PATH):
    return np.load(path)["ref"] if Path(path).exists() else None


def stimulus(frame, geom, buttons, refs, light_on, cam=None, cove_ref=None):  # cove_ref: the empty-camera reference for `cam` (1C or 4B)
    """What the fly's eye is shown: the frame, except each hallway patch shows how much it differs from the EMPTY hallway while
    that light is on (black otherwise). Anything standing in the hallway - Bonnie, Chica, Foxy - is a bright blob on black, so
    the fly's response no longer depends on which animatronic it is. light_on None = show the raw frame (monitor up),
    except cam 1C / 4B with a reference: that picture shows how much it differs from the empty view (1C: curtain closed, 4B: nobody there)."""
    if light_on is None and cam in ("1C", "4B") and cove_ref is not None:
        g = cove_gray(frame)
        if g.shape == cove_ref.shape:
            y0, y1, x0, x1 = cove_box(frame)
            out = frame.copy()
            out[y0:y1, x0:x1] = np.clip(np.abs(g - cove_ref) * config.COVE_GAIN, 0, 255)[..., None].astype(frame.dtype)
            return out
        return frame
    if refs is None or light_on is None:
        return frame
    out = frame.copy()
    for s in "LR":
        x0, y0, x1, y1 = roi_box(frame, geom, "hall_" + s, buttons)
        g = frame[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
        ref = refs[s]
        d = np.zeros_like(g)
        if light_on.get(s) is True and g.shape == ref.shape and g.mean() >= LIT_MIN:
            d = np.clip(np.abs(g - ref) * DIFF_GAIN, 0, 255)
        out[y0:y1, x0:x1] = d[..., None].astype(frame.dtype)
    return out


class Readout:
    """danger = sigmoid(w . ((features - side baseline) / sd) + b); fitted on the labeled corpus."""

    def __init__(self, path=READOUT_PATH):
        z = np.load(path)
        self.types = [t for t in z["types"]]
        self.w, self.b, self.sd = z["w"], float(z["b"]), z["sd"]
        self.base = {"L": z["baseL"], "R": z["baseR"]}

    def z(self, side, feats):
        return (np.asarray(feats) - self.base[side]) / self.sd

    def score(self, side, feats):
        return float(_sigmoid(self.z(side, feats) @ self.w + self.b))


class CoveReadout:
    """cove = sigmoid(w . ((cam_feat[use] - mean) / sd) + b): 'Foxy is out of his curtain / gone'. Fitted by fit_cove."""

    def __init__(self, path=COVE_PATH):
        z = np.load(path)
        self.use, self.w, self.b, self.mean, self.sd = z["use"], z["w"], float(z["b"]), z["mean"], z["sd"]
        self.threshold, self.mode = float(z["threshold"]), str(z["mode"])

    def score(self, cam_feat):
        x = (np.asarray(cam_feat)[self.use] - self.mean) / self.sd
        return float(_sigmoid(x @ self.w + self.b))


def gone_features(act, xy, grid=GONE_GRID):
    """act (types, hexals over CAM_VIEW), xy = (x, y) of those hexals -> per-type mean activity in each grid cell (types * nx * ny)."""
    nx, ny = grid
    x, y = ((v - v.min()) / (np.ptp(v) + 1e-9) for v in xy)
    cell = np.minimum((x * nx).astype(int), nx - 1) * ny + np.minimum((y * ny).astype(int), ny - 1)
    return np.concatenate([np.stack([a[cell == c].mean() for c in range(nx * ny)]) for a in act])


class CoveGoneReadout:
    """cove_gone = sigmoid(w . ((gone_features - mean) / sd) + b): 'the cove is EMPTY, Foxy is running' (stage 4). Fitted by fit_cove_gone."""

    def __init__(self, path=COVE_GONE_PATH):
        z = np.load(path)
        self.w, self.b, self.mean, self.sd, self.threshold = z["w"], float(z["b"]), z["mean"], z["sd"], float(z["threshold"])

    def score(self, act, xy):
        return float(_sigmoid((gone_features(act, xy) - self.mean) / self.sd @ self.w + self.b))


class FlyBrain:
    """Runs the network on the latest frames in its own thread. snapshot() is cheap and thread-safe."""

    def __init__(self, geom, buttons, readout_path=READOUT_PATH, perception=None, view=True, diff=True):
        self.geom, self.buttons = geom, buttons
        self.readout = Readout(readout_path) if Path(readout_path).exists() else None
        self.refs = load_refs() if diff else None
        self.cove = CoveReadout() if Path(COVE_PATH).exists() else None
        self.cove_ref = load_cove_ref() if (diff and (self.cove is None or self.cove.mode == "diff")) else None
        self.cam4b = CoveReadout(CAM4B_PATH) if Path(CAM4B_PATH).exists() else None
        self.gone = CoveGoneReadout() if Path(COVE_GONE_PATH).exists() else None
        self.cam4b_ref = load_cove_ref(CAM4B_REF_PATH) if (diff and (self.cam4b is None or self.cam4b.mode == "diff")) else None
        self._p = perception
        self._q, self._cv, self._lock = deque(maxlen=2), threading.Condition(), threading.Lock()
        self._snap, self.view = None, view
        self.ms, self.error = 0.0, None
        threading.Thread(target=self._guarded, daemon=True).start()

    def submit(self, frame, t, light_on=None, cam=None):
        with self._cv:
            self._q.append((frame, t, light_on, cam))
            self._cv.notify()

    def snapshot(self):
        with self._lock:
            return self._snap

    def _guarded(self):
        try:
            self._run()
        except BaseException as e:  # a dead brain must not leave the bot acting on its last verdict
            self.error = e

    def _run(self):
        import torch
        from PIL import Image
        torch.set_num_threads(4)  # leave cores for capture + the control loop
        if self._p is None:
            from perception import Perception
            self._p = Perception()
        p = self._p
        c = p.net.connectome
        idx = {t: np.asarray(c.nodes.layer_index[t][:]) for t in set(VIEW_TYPES) | set(READOUT_TYPES)}
        xg, yg = hex_frame_coords(p)
        masks, ix, cam_ix, regions, ema, scale = None, None, None, None, {}, {t: 1e-3 for t in VIEW_TYPES}
        grid = np.stack([np.asarray(c.nodes.u[:])[idx["R1"]], np.asarray(c.nodes.v[:])[idx["R1"]]], 1).tolist()  # constant: serialise once
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                frame, t, light_on, cam = self._q.popleft()
            t0 = time.time()
            if masks is None:
                masks = region_masks(frame.shape, self.geom, self.buttons, xg, yg)
                ix = {s: [idx[ty][masks[s]] for ty in READOUT_TYPES] for s in "LR"}  # flat hexal indices per side and type
                regions = {s: np.flatnonzero(masks[s]).tolist() for s in "LR"}
                cm = (xg >= CAM_VIEW[0]) & (xg <= CAM_VIEW[1]) & (yg >= CAM_VIEW[2]) & (yg <= CAM_VIEW[3])
                cam_ix = [idx[ty][cm] for ty in READOUT_TYPES] + [np.concatenate([idx[ty][cm] for ty in names]) for names in (T4, T5)]
                cam_xy = (xg[cm], yg[cm])
            p.step(stimulus(frame, self.geom, self.buttons, self.refs, light_on, cam, self.cam4b_ref if cam == "4B" else self.cove_ref))
            act = (p.state.nodes.activity - p.baseline)[0].numpy()
            feats = {s: np.array([act[i].mean() for i in ix[s]]) for s in "LR"}
            cam_feat = [float(act[i].mean()) for i in cam_ix[:4]] + [float(np.abs(act[i]).mean()) for i in cam_ix[4:]]  # + motion energy
            danger = {s: (self.readout.score(s, feats[s]) if self.readout else None) for s in "LR"}
            cove = self.cove.score(cam_feat) if (self.cove and cam == "1C" and light_on is None) else None
            cam4b = self.cam4b.score(cam_feat) if (self.cam4b and cam == "4B" and light_on is None) else None
            gone = self.gone.score(np.stack([act[i] for i in cam_ix[:4]]), cam_xy) if (self.gone and cam == "1C" and light_on is None) else None
            snap = {"t": t, "danger": danger, "cam_feat": cam_feat, "cam": cam, "cove": cove, "cam4b": cam4b, "cove_gone": gone, "feats": {s: feats[s].tolist() for s in "LR"}}
            if self.readout:
                snap["contrib"] = {s: (self.readout.z(s, feats[s]) * self.readout.w).tolist() for s in "LR"}
            if self.view:
                maps = {}
                for ty in VIEW_TYPES:
                    a = act[idx[ty]]
                    ema[ty] = a.copy() if ty not in ema else 0.98 * ema[ty] + 0.02 * a  # show change from each cell's own average
                    d = a - ema[ty]
                    scale[ty] = max(0.97 * scale[ty], float(np.percentile(np.abs(d), 99)), 1e-6)
                    maps[ty] = np.clip(np.round(d / scale[ty] * 127), -127, 127).astype(int).tolist()
                buf = io.BytesIO()
                im = Image.fromarray(np.ascontiguousarray(frame))
                im.resize((JPEG_W, int(JPEG_W * im.height / im.width))).save(buf, "JPEG", quality=65)
                snap.update(maps=maps, jpeg=base64.b64encode(buf.getvalue()).decode(), grid=grid, regions=regions)
            self.ms = (time.time() - t0) * 1000
            snap["ms"] = round(self.ms, 1)
            with self._lock:
                self._snap = snap


class FlyHallway:
    """Drop-in for readers.hallway.HallwayDetector: hall = the fly's danger score, only while that light is on.
    The game flickers the hallway light, which shows up as single frames of ~0 danger, so a side reports the max over its
    last HALL_SMOOTH_FRAMES lit frames."""

    def __init__(self, brain, smooth=None):
        self.brain, self.smooth = brain, smooth or config.HALL_SMOOTH_FRAMES
        self.hist = {"L": deque(maxlen=self.smooth), "R": deque(maxlen=self.smooth)}
        self.cam_snap_t = None  # t of the last snapshot whose cove/cam4b score was handed out

    STALE_S = 1.5  # a verdict older than this (brain too slow / stuck) is not used

    def read(self, frame, t, light_on=None):
        if self.brain.error is not None:
            raise RuntimeError(f"fly brain thread died: {self.brain.error!r}")
        self.brain.submit(frame, t, light_on or {})
        snap = self.brain.snapshot()
        if snap and t - snap["t"] > self.STALE_S:
            snap = None
        lo = light_on or {}
        hall = {}
        for s in "LR":
            if snap and lo.get(s) is True and snap["danger"][s] is not None:
                self.hist[s].append(snap["danger"][s])
                hall[s] = max(self.hist[s])
            else:
                self.hist[s].clear()
                hall[s] = None
        return Readings(t, hall=hall)


    def cam_read(self, frame, t, cam=None):
        """Camera view: no light gating. cove = the fly's cove score, only for a fresh snapshot computed while cam 1C was shown."""
        self.brain.submit(frame, t, None, cam)
        snap = self.brain.snapshot()
        for s in "LR":
            self.hist[s].clear()
        fresh = snap if snap and t - snap["t"] <= 0.5 else None
        # each fly verdict once: the loop outruns the fly, and a repeated snapshot counted as a second confirming frame
        # (live 2026-10-05 N3: one static transition frame read 0.98 twice -> foxy_close, left door shut 41 s for nothing)
        new = fresh if fresh and fresh["t"] != self.cam_snap_t else None
        if new:
            self.cam_snap_t = new["t"]
        return Readings(t, cam_feat=fresh["cam_feat"] if fresh else None,
                        cove=new["cove"] if new and cam == "1C" and new.get("cam") == "1C" else None,
                        cam4b=new["cam4b"] if new and cam == "4B" and new.get("cam") == "4B" else None,
                        cove_gone=new.get("cove_gone") if new and cam == "1C" and new.get("cam") == "1C" else None)


class Tee:
    """--perception pixel with the live page on: feed the brain too (for display) but let the pixel detector judge."""

    def __init__(self, brain, inner):
        self.brain, self.inner = brain, inner

    def read(self, frame, t, light_on=None):
        self.brain.submit(frame, t)
        return self.inner.read(frame, t, light_on)


def fit(sessions=("n1_a", "n2_a"), out=READOUT_PATH):
    """Fit the readout from the fly's regional activity over the labeled lit-hallway frames. Features come from
    scripts/fly_danger_corpus.py (the fly run continuously over a corpus session with the same input it gets live:
    frame minus empty hallway). Positives exist only on the left (Bonnie), so the weights come from the left; each side gets its
    own empty-hallway baseline. Because the input is a difference from empty, the same weights are meant to carry over to the
    right hallway (and other animatronics), but that is unverified until one shows up there."""
    import csv

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rows = []
    for sess in sessions:
        z = np.load(f"data/features/fly_diff_{sess}.npz")
        for r in csv.DictReader(open(f"{config.CORPUS_DIR}/{sess}/labels.csv")):
            i = int(r["frame"])
            for k, s in enumerate("LR"):
                if r.get("hall_" + s) in ("0", "1") and z["light"][i, k] == 1 and z["door"][i, k] != 1 and not np.isnan(z["feats"][i, k, 0]):
                    rows.append((s, int(r["hall_" + s]), (sess, i // 300), z["feats"][i, k].astype(float)))
    base = {s: np.mean([r[3] for r in rows if r[0] == s and r[1] == 0], axis=0) for s in "LR"}
    L = [r for r in rows if r[0] == "L"]
    sd = np.std([r[3] - base["L"] for r in L], axis=0) + 1e-9
    X, y, blk = np.stack([(r[3] - base["L"]) / sd for r in L]), np.array([r[1] for r in L]), [r[2] for r in L]
    cv = np.zeros(len(y))
    for k in set(blk):  # hold out whole episodes (blocks of 300 frames)
        te = np.array([b == k for b in blk])
        if y[~te].min() != y[~te].max():
            cv[te] = LogisticRegression(C=1.0).fit(X[~te], y[~te]).decision_function(X[te])
    m = LogisticRegression(C=1.0).fit(X, y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, types=np.array(READOUT_TYPES), w=m.coef_[0], b=m.intercept_[0], sd=sd, baseL=base["L"], baseR=base["R"])
    return {"left_occupied": int(y.sum()), "left_empty": int((y == 0).sum()), "right_empty": sum(1 for r in rows if r[0] == "R"),
            "heldout_episode_auc": float(roc_auc_score(y, cv)), "weights": dict(zip(READOUT_TYPES, m.coef_[0].round(2).tolist()))}


def cove_labels(sess, cam="1C"):
    """-> list of (frame, stage, look_id). 1C: <session>/labels_cove.csv (stage 1-4). 4B: labels_4b.csv, stage 1 = empty and 3 = Freddy or
    Chica present (x rows dropped), so the same fit code works for both cameras (positive = stage >= 2)."""
    import csv
    if cam == "1C":
        return [(int(r["frame"]), int(r["stage"]), int(r["look_id"])) for r in csv.DictReader(open(f"{config.CORPUS_DIR}/{sess}/labels_cove.csv"))
                if r["stage"] in "1234"]
    return [(int(r["frame"]), 3 if "1" in (r["freddy"], r["chica"]) else 1, int(r["look_id"]))
            for r in csv.DictReader(open(f"{config.CORPUS_DIR}/{sess}/labels_4b.csv")) if r["freddy"] != "x"]


def build_cove_ref(sessions, out=None, cam="1C"):
    """Per-pixel mean gray of the camera picture over the empty labelled frames (1C: stage 1, 4B: nobody there)."""
    from PIL import Image
    out = out or CAM_PATHS[cam][0]
    acc = []
    for sess in sessions:
        for i, stage, _ in cove_labels(sess, cam):
            if stage == 1:
                acc.append(cove_gray(np.asarray(Image.open(f"{config.CORPUS_DIR}/{sess}/frames/{i:06d}.png").convert("RGB"))))
    ref = np.mean(acc, axis=0).astype(np.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ref=ref)
    return len(acc)


COVE_FEATURES = {"all": [0, 1, 2, 3, 4, 5], "no_motion": [0, 1, 2, 3]}  # indices into cam_feat (R1, L1, L2, Mi1 means, T4/T5 energy)


def fit_cove(sessions, mode="diff", features="all", out=None, write=True, min_stage=2, cam="1C"):
    """Fit a camera readout from data/features/fly_<cove|cam4b>_<mode>_<sess>.npz (scripts/fly_danger_corpus.py --cove / --cam4b).
    Label = stage >= min_stage (1C: 2 = out of his curtain, 4 = gone from the cove; 4B: occupied), cross-validated holding out whole
    looks; threshold = lowest held-out score keeping empty/stage-1 false alarms <= 2%."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    out = out or CAM_PATHS[cam][1]
    tag = "cove" if cam == "1C" else "cam4b"
    X, y, stage, grp = [], [], [], []
    for sess in sessions:
        z = np.load(f"data/features/fly_{tag}_{mode}_{sess}.npz")
        pos = {int(fr): k for k, fr in enumerate(z["frames"])}
        for i, st, look in cove_labels(sess, cam):
            if i in pos and not np.isnan(z["feats"][pos[i]]).any():
                X.append(z["feats"][pos[i]]); y.append(int(st >= min_stage)); stage.append(st); grp.append((sess, look))
    use = np.array(COVE_FEATURES[features])
    X, y, stage = np.array(X)[:, use], np.array(y), np.array(stage)
    mean, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / sd
    cv = np.zeros(len(y))
    for g in set(grp):
        te = np.array([k == g for k in grp])
        if y[~te].min() != y[~te].max():
            cv[te] = LogisticRegression(C=1.0).fit(Z[~te], y[~te]).decision_function(Z[te])
    sc = _sigmoid(cv)
    neg = np.sort(sc[y == 0])
    thr = float(neg[int(np.ceil(0.98 * len(neg))) - 1]) + 1e-6  # <= 2% of the empty frames score above it
    res = {"cam": cam, "mode": mode, "features": features, "n": len(y), "positives": int(y.sum()), "heldout_look_auc": float(roc_auc_score(y, sc)),
           "threshold": thr, "stage1_false_alarm": float((sc[y == 0] > thr).mean()), "hit_rate": float((sc[y == 1] > thr).mean()),
           "hit_by_stage": {int(s): float((sc[stage == s] > thr).mean()) for s in (2, 3, 4) if (stage == s).any()},
           "false_alarm_by_stage": {int(s): float((sc[stage == s] > thr).mean()) for s in (1, 2, 3) if (stage == s).any()}}
    if write:
        m = LogisticRegression(C=1.0).fit(Z, y)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, use=use, w=m.coef_[0], b=m.intercept_[0], mean=mean, sd=sd, threshold=thr, mode=mode)
    return res


def fit_cove_gone(sessions, C=0.1, q=0.98, out=COVE_GONE_PATH, write=True):
    """Fit the 'cove empty' readout from data/features/fly_cove_grid_<sess>.npz (scripts/fly_danger_corpus.py --cove-grid). Label = stage 4.
    Cross-validated holding out whole looks; threshold = held-out score that keeps (1 - q) of the stage 1-3 frames above it."""
    from sklearn.linear_model import LogisticRegression
    X, st, grp = [], [], []
    for sess in sessions:
        z = np.load(f"data/features/fly_cove_grid_{sess}.npz")
        X += [gone_features(a, (z["xg"], z["yg"])) for a in z["act"]]
        st += list(z["stage"]); grp += [(sess, int(k)) for k in z["look"]]
    X, st = np.array(X), np.array(st)
    y = (st == 4).astype(int)
    mean, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / sd
    cv = np.zeros(len(y))
    for g in set(grp):
        te = np.array([k == g for k in grp])
        cv[te] = LogisticRegression(C=C, max_iter=2000).fit(Z[~te], y[~te]).decision_function(Z[te])
    sc = _sigmoid(cv)
    neg = np.sort(sc[y == 0])
    thr = float(neg[int(np.ceil(q * len(neg))) - 1]) + 1e-6
    looks = {g for g, v in zip(grp, y) if v}
    res = {"n": len(y), "gone_frames": int(y.sum()), "gone_looks": len(looks), "threshold": thr,
           "hit_rate": float((sc[y == 1] > thr).mean()), "false_alarm_by_stage": {int(s): float((sc[st == s] > thr).mean()) for s in (1, 2, 3)}}
    if write:
        m = LogisticRegression(C=C, max_iter=2000).fit(Z, y)
        np.savez(out, w=m.coef_[0], b=m.intercept_[0], mean=mean, sd=sd, threshold=thr, grid=np.array(GONE_GRID))
    return res


if __name__ == "__main__":
    print(fit())
