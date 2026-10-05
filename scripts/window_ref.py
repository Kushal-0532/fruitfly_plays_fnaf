"""Right hall = the right WINDOW: Chica shows up in the window beside the right doorway, not in it (live 2026-10-05 N3 round 5: Chica
in the lit window 419-421 s, fly hall_R 0.07 because its patch covered the empty doorway). After hall_R in calibration.json points at
the window, rebuild from recordings (no labels needed, the button reader says when the right light is on):
  1. refR in data/templates/hall_ref.npz = mean lit, empty window (frames far from the median = somebody there, dropped)
  2. baseR in data/templates/fly_readout.npz = the fly's mean right-patch activity on those empty frames (weights stay the left's)
  3. prints every lit-window frame the fly now scores >= 0.5, per session, so the Chica frames can be checked
Old files are kept once as *.doorway.npz.
usage: python -m scripts.window_ref data/corpus/n3_readout_*"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
from fly_brain import READOUT_PATH, REFS_PATH, FlyBrain, _sigmoid
from geometry import load_buttons, roi_box
from readers.buttons import ButtonReader
from readers.digits import _geom
from readers.hallway import LIT_MIN

OUTLIER = 3.0  # a lit frame whose MSE to the median window is > OUTLIER x the median MSE has somebody in it


def lit_frames(sessions, buttons):
    """-> [(session, frame index, lit gray window)] for frames with the right light on, right door open, monitor down."""
    out = []
    for d in sessions:
        meta = json.loads((d / "meta.json").read_text())
        geom = _geom(meta)
        btn = ButtonReader(geom, buttons)
        for i in range(len(meta["times"])):
            f = np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
            r = btn.read(f, 0)
            if r.monitor_up or (r.light_on or {}).get("R") is not True or (r.door_closed or {}).get("R") is not False:
                continue
            x0, y0, x1, y1 = roi_box(f, geom, "hall_R", buttons)
            g = f[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
            if g.mean() >= LIT_MIN:
                out.append((d, i, g))
        print(d.name, "lit right-window frames so far:", len(out), flush=True)
    return out


def build_ref(lit):
    shape = max({g.shape for _, _, g in lit}, key=lambda s: sum(g.shape == s for _, _, g in lit))  # the common ROI size
    lit = [x for x in lit if x[2].shape == shape]
    med = np.median([g for _, _, g in lit], axis=0)
    mse = np.array([((g - med) ** 2).mean() for _, _, g in lit])
    keep = mse <= OUTLIER * np.median(mse)
    ref = np.mean([g for (_, _, g), k in zip(lit, keep) if k], axis=0)
    e = [float(((g - ref) ** 2).mean()) for (_, _, g), k in zip(lit, keep) if k]
    old = dict(np.load(REFS_PATH))
    backup = Path(REFS_PATH).with_suffix(".doorway.npz")
    if not backup.exists():
        shutil.copy(REFS_PATH, backup)
    np.savez(REFS_PATH, **{**old, "refR": ref, "noiseR": np.array((float(np.mean(e)), float(np.std(e)) or 1.0))})
    print(f"refR {shape}: {keep.sum()} empty frames, {(~keep).sum()} dropped as occupied", flush=True)
    return {(d.name, i) for (d, i, _), k in zip(lit, keep) if k}


def fly_pass(sessions, buttons, lit):
    """Run the fly over each lit run (from 5 frames before) -> {(session, i): right-patch features}."""
    want = {}
    for d, i, _ in lit:
        want.setdefault(d, set()).add(i)
    brain, feats = None, {}
    for k, d in enumerate(sessions):
        meta = json.loads((d / "meta.json").read_text())
        geom = _geom(meta)
        if brain is None:
            brain = FlyBrain(geom, buttons, view=False)  # one network per process (flyvis); recordings share the window size
        btn = ButtonReader(geom, buttons)
        todo = sorted({j for i in want.get(d, ()) for j in range(max(i - 5, 0), i + 1)})
        for i in todo:
            f = np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
            r = btn.read(f, 0)
            t = k * 1e5 + i / 10
            brain.submit(f, t, None if r.monitor_up else (r.light_on or {}))
            while (s := brain.snapshot()) is None or abs(s["t"] - t) > 1e-6:
                time.sleep(0.005)
            if i in want[d]:
                feats[(d.name, i)] = np.array(s["feats"]["R"], float)
        print(d.name, "fly done", flush=True)
    return feats


def main(paths):
    sessions = [Path(p) for p in paths]
    buttons = load_buttons(config.CALIBRATION_PATH)
    lit = lit_frames(sessions, buttons)
    empty = build_ref(lit)
    feats = fly_pass(sessions, buttons, lit)
    z = dict(np.load(READOUT_PATH))
    backup = Path(READOUT_PATH).with_suffix(".doorway.npz")
    if not backup.exists():
        shutil.copy(READOUT_PATH, backup)
    z["baseR"] = np.mean([v for k, v in feats.items() if k in empty], axis=0)
    np.savez(READOUT_PATH, **z)
    score = lambda v: float(_sigmoid(((v - z["baseR"]) / z["sd"]) @ z["w"] + z["b"]))
    for d in sessions:
        times = json.loads((d / "meta.json").read_text())["times"]
        sc = {i: score(v) for (sn, i), v in feats.items() if sn == d.name}
        hot = [(round(times[i], 1), round(s, 2)) for i, s in sorted(sc.items()) if s >= 0.5]
        emp = [s for i, s in sc.items() if (d.name, i) in empty]
        print(f"{d.name}: {len(sc)} lit frames, empty max {max(emp, default=0):.2f}, >= 0.5 at (t, score): {hot}")


if __name__ == "__main__":
    main(sys.argv[1:])
