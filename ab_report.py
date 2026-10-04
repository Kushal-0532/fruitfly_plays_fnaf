"""Phase 19: pixel vs fly hallway detector on the labeled corpus, AUC with bootstrap intervals.
usage: python ab_report.py data/corpus --out logs/ab_report.json"""
import argparse
import json
from pathlib import Path

import numpy as np

import config
from geometry import load_buttons
from readers import hallway as H
from readers.digits import _geom
from shadow import N_BURST, FlyScorer, hallway_diff

MARGIN = 0.05  # the fly gets a vote only if its AUC lower bound >= pixel AUC upper bound - MARGIN


def auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def boot_ci(scores, labels, n=1000, seed=0):
    rng, s, y = np.random.default_rng(seed), np.asarray(scores), np.asarray(labels)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(auc(s[i], y[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compare(pixel, fly, labels):
    p, f = auc(pixel, labels), auc(fly, labels)
    (plo, phi), (flo, fhi) = boot_ci(pixel, labels), boot_ci(fly, labels)
    return {"pixel_auc": p, "pixel_ci": [plo, phi], "fly_auc": f, "fly_ci": [flo, fhi], "promote": flo >= phi - MARGIN}


def bursts(session, side, buttons, ref):
    """Light-onset bursts: the dark frame before the hallway lights up plus the next N_BURST lit frames.
    Label = hall label of a labeled frame inside the burst (labels are every 5th frame, so most bursts have one)."""
    from PIL import Image
    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    geom = _geom(meta)
    import csv
    labels = {int(r["frame"]): r["hall_" + side] for r in csv.DictReader(open(d / "labels.csv")) if r["hall_" + side] in ("0", "1")}
    n = len(meta["times"])
    grays, mse = [], []
    for i in range(n):
        g = H._gray(np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB")), geom, side, buttons)
        grays.append(g)
    lit = [g.mean() >= H.LIT_MIN for g in grays]
    out = []
    for i in range(1, n - N_BURST):
        if not lit[i - 1] and all(lit[i:i + N_BURST]):
            ys = {labels[j] for j in range(i, i + N_BURST) if j in labels}
            if len(ys) == 1:
                seq = grays[i - 1:i + N_BURST]
                out.append({"label": int(ys.pop()), "diffs": [hallway_diff(g, ref) for g in seq],
                            "pixel": float(((grays[i + N_BURST - 1] - ref) ** 2).mean())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", default="logs/ab_report.json")
    a = ap.parse_args(argv)
    buttons = load_buttons(config.CALIBRATION_PATH)
    refs = np.load("data/templates/hall_ref.npz")
    scorer, report = FlyScorer(), {}
    for side in "LR":
        bs = []
        for s in sorted(p for p in Path(a.corpus).iterdir() if (p / "meta.json").exists()):
            bs += bursts(s, side, buttons, refs["ref" + side])
        y = [b["label"] for b in bs]
        report[side] = {"bursts": len(bs), "occupied": int(sum(y))}
        if 0 < sum(y) < len(y):
            fly = [sum(scorer.score_burst(b["diffs"]).values()) for b in bs]
            report[side].update(compare([b["pixel"] for b in bs], fly, y))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
