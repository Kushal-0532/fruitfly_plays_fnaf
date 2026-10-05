"""Run the fly over every frame of a corpus session; save per-frame danger scores + the reader state. Offline, ~60 ms/frame.
usage: python scripts/fly_danger_corpus.py [--cove | --cam4b | --cove-grid] data/corpus/n2_a"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
from fly_brain import FlyBrain
from geometry import load_buttons
from readers.buttons import ButtonReader
from readers.digits import _geom


def cove(session, raw=False, cam="1C"):
    """Phase 34: the fly over every look in labels_cove.csv (continuously, from 1 s before the first labelled frame of a look),
    with the cam-1C stimulus (diff vs the stage-1 reference, or the raw frame). -> data/features/fly_cove_<mode>_<session>.npz"""
    from fly_brain import cove_labels
    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    geom, buttons = _geom(meta), load_buttons(config.CALIBRATION_PATH)
    brain = FlyBrain(geom, buttons, view=False, diff=not raw)
    brain.cove = None  # extraction must not depend on a previously fitted readout
    looks = {}
    for i, _, look in cove_labels(d.name, cam):
        looks.setdefault(look, []).append(i)
    frames, look_of, feats = [], [], []
    for look, fr in sorted(looks.items()):
        for i in range(max(min(fr) - 10, 0), max(fr) + 1):
            f = np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
            brain.submit(f, i / 10, None, cam)
            while True:
                s = brain.snapshot()
                if s is not None and abs(s["t"] - i / 10) < 1e-6:
                    break
                time.sleep(0.005)
            frames.append(i); look_of.append(look); feats.append(s["cam_feat"])
        print(session, "look", look, len(frames), flush=True)
    np.savez(f"data/features/fly_{'cove' if cam == '1C' else 'cam4b'}_{'raw' if raw else 'diff'}_{d.name}.npz", frames=np.array(frames), look=np.array(look_of), feats=np.array(feats, np.float32))


def cove_grid(session):
    """The fly's per-hexal activity (READOUT_TYPES over CAM_VIEW) on every labelled 1C look, run continuously from 10 frames before
    the look -> data/features/fly_cove_grid_<session>.npz, for fly_brain.fit_cove_gone."""
    import fly_brain as F
    from perception import Perception
    p = Perception()
    idx = {ty: np.asarray(p.net.connectome.nodes.layer_index[ty][:]) for ty in F.READOUT_TYPES}
    xg, yg = F.hex_frame_coords(p)
    cm = (xg >= F.CAM_VIEW[0]) & (xg <= F.CAM_VIEW[1]) & (yg >= F.CAM_VIEW[2]) & (yg <= F.CAM_VIEW[3])
    ref = F.load_cove_ref()
    d = Path(session)
    looks = {}
    for i, st, look in F.cove_labels(d.name):
        looks.setdefault(look, {})[i] = st
    frames, look_of, stage, acts = [], [], [], []
    for look, lab in sorted(looks.items()):
        for i in range(max(min(lab) - 10, 0), max(lab) + 1):
            p.step(F.stimulus(np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB")), None, None, None, None, "1C", ref))
            if i in lab:
                a = (p.state.nodes.activity - p.baseline)[0].numpy()
                acts.append(np.stack([a[idx[ty][cm]] for ty in F.READOUT_TYPES]))
                frames.append(i); look_of.append(look); stage.append(lab[i])
    np.savez(f"data/features/fly_cove_grid_{d.name}.npz", frames=frames, look=look_of, stage=stage, act=np.array(acts, np.float32),
             xg=xg[cm], yg=yg[cm])
    print(session, len(frames), "frames")


def main(session):
    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    geom, buttons = _geom(meta), load_buttons(config.CALIBRATION_PATH)
    brain = FlyBrain(geom, buttons, view=False)
    btn = ButtonReader(geom, buttons)
    n = len(meta["times"])
    out = {"danger": np.full((n, 2), np.nan, np.float32), "light": np.zeros((n, 2), np.int8), "monitor": np.zeros(n, np.int8),
           "door": np.full((n, 2), -1, np.int8), "feats": np.full((n, 2, 4), np.nan, np.float32)}
    t0 = time.time()
    for i in range(n):
        f = np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
        r = btn.read(f, 0)
        brain.submit(f, i / 10, None if r.monitor_up else (r.light_on or {}))
        while True:  # wait for the worker to finish this frame, so every frame gets its own result
            s = brain.snapshot()
            if s is not None and abs(s["t"] - i / 10) < 1e-6:
                break
            time.sleep(0.005)
        out["monitor"][i] = bool(r.monitor_up)
        for k, s_ in enumerate("LR"):
            li = (r.light_on or {}).get(s_)
            out["light"][i, k] = -1 if li is None else int(li)
            out["door"][i, k] = -1 if (r.door_closed or {}).get(s_) is None else int(r.door_closed[s_])
            out["danger"][i, k] = s["danger"][s_]
            out["feats"][i, k] = s["feats"][s_]
        if i % 500 == 0:
            print(session, i, n, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(f"data/features/fly_diff_{d.name}.npz", **out)


if __name__ == "__main__":
    if sys.argv[1] == "--cove-grid":
        cove_grid(sys.argv[-1])
    elif sys.argv[1] in ("--cove", "--cam4b"):
        cove(sys.argv[-1], raw="--raw" in sys.argv, cam="1C" if sys.argv[1] == "--cove" else "4B")
    else:
        main(sys.argv[1])
