"""Phase 10: label tooling. template -> labels.csv, sheet -> contact sheets, check [--ready] -> validate / write READY."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import config
from geometry import REQUIRED_BUTTONS, Uncalibrated, load_buttons

COLUMNS = ["frame", "hour", "power_pct", "usage", "monitor_up", "door_L", "door_R", "light_L", "light_R",
           "cam", "foxy_stage", "hall_L", "hall_R"]
CAMS = [n[4:] for n in REQUIRED_BUTTONS if n.startswith("cam_") and n != "cam_map"]
BOOLS = ("monitor_up", "door_L", "door_R", "light_L", "light_R", "hall_L", "hall_R")
FOXY_REQUIRED = ("active_night1", "side")


def _num(v, lo, hi, integer):
    x = int(v) if integer else float(v)
    if not lo <= x <= hi:
        raise ValueError
    return x


def foxy_info(corpus_dir):
    """-> (stage numbers, missing-key problems). File absent -> (0..6, ['foxy_stages.txt missing'])."""
    p = Path(corpus_dir) / "foxy_stages.txt"
    if not p.exists():
        return set(range(7)), ["foxy_stages.txt missing"]
    keys = {}
    for line in p.read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            keys[k.strip().lower()] = v.strip()
    stages = {int(m.group(1)) for k in keys if (m := re.fullmatch(r"stage (\d+)", k))}
    problems = [f"foxy_stages.txt missing key {k}" for k in FOXY_REQUIRED if not keys.get(k)]
    if not 2 <= len(stages) <= 6:
        problems.append("foxy_stages.txt needs 2-6 'stage N:' lines")
    return stages or set(range(7)), problems


def read_labels(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def validate(path, foxy_stages=None):
    """-> (errors, rows). Blank cells are fine (unlabeled)."""
    rows, cols = read_labels(path)
    errors = [f"missing column {c}" for c in COLUMNS if c not in cols]
    if errors:
        return errors, rows
    stages = foxy_stages if foxy_stages is not None else set(range(7))
    for i, row in enumerate(rows, 2):
        for c in COLUMNS[1:]:
            v = (row[c] or "").strip()
            if not v:
                continue
            try:
                if c == "hour":
                    _num(v, 0, 6, True)
                elif c == "power_pct":
                    _num(v, 0, 100, False)
                elif c == "usage":
                    _num(v, 1, 4, True)
                elif c in BOOLS:
                    if v not in ("0", "1"):
                        raise ValueError
                elif c == "cam":
                    if v not in CAMS:
                        raise ValueError
                elif c == "foxy_stage":
                    if int(v) not in stages:
                        raise ValueError
            except ValueError:
                errors.append(f"{path.name} line {i}: bad {c} value {v!r}")
    return errors, rows


def counts(rows):
    out = {}
    for c in COLUMNS[1:]:
        for row in rows:
            v = (row.get(c) or "").strip()
            if v:
                out.setdefault(c, {}).setdefault(v, 0)
                out[c][v] += 1
    return out


def make_template(session_dir, every=10):
    d = Path(session_dir)
    n = len(json.loads((d / "meta.json").read_text())["times"])
    with open(d / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(0, n, every):
            w.writerow([i] + [""] * (len(COLUMNS) - 1))
    return d / "labels.csv"


def make_sheets(session_dir, cols=4, rows=3, tile_w=480):
    """One contact sheet per cols*rows labeled frames -> sheets/NN.png (frame index drawn on each tile)."""
    from PIL import Image, ImageDraw
    d = Path(session_dir)
    idx = [int(r["frame"]) for r in read_labels(d / "labels.csv")[0]]
    (d / "sheets").mkdir(exist_ok=True)
    out, per = [], cols * rows
    for s in range(0, len(idx), per):
        chunk = idx[s:s + per]
        tiles = []
        for i in chunk:
            im = Image.open(d / "frames" / f"{i:06d}.png").convert("RGB")
            im = im.resize((tile_w, round(im.height * tile_w / im.width)))
            ImageDraw.Draw(im).text((4, 4), str(i), fill=(255, 255, 0))
            tiles.append(im)
        th = tiles[0].height
        sheet = Image.new("RGB", (cols * tile_w, -(-len(tiles) // cols) * th))
        for k, im in enumerate(tiles):
            sheet.paste(im, ((k % cols) * tile_w, (k // cols) * th))
        out.append(d / "sheets" / f"{s // per:02d}.png")
        sheet.save(out[-1])
    return out


def ready_failures(corpus_dir, calibration_path=config.CALIBRATION_PATH):
    """All minimums for READY; returns a list of failure strings (empty == ready)."""
    cd, fails, allrows = Path(corpus_dir), [], []
    sessions = [p for p in cd.iterdir() if (p / "meta.json").exists()] if cd.exists() else []
    stages, foxy_problems = foxy_info(cd)
    for s in sessions:
        if (s / "labels.csv").exists():
            errs, rows = validate(s / "labels.csv", stages)
            fails += errs
            allrows += rows
    if len(sessions) < 3:
        fails.append(f"need >= 3 sessions, have {len(sessions)}")
    c = counts(allrows)
    for side in "LR":
        k = c.get(f"hall_{side}", {})
        if sum(k.values()) < 100:
            fails.append(f"hall_{side}: need >= 100 labeled rows, have {sum(k.values())}")
        for cls in "01":
            if k.get(cls, 0) < 20:
                fails.append(f"hall_{side}={cls}: need >= 20 rows, have {k.get(cls, 0)}")
    both = sum(1 for r in allrows if (r.get("hour") or "").strip() and (r.get("power_pct") or "").strip())
    if both < 100:
        fails.append(f"hour+power_pct: need >= 100 rows, have {both}")
    for cls in "01":
        if c.get("monitor_up", {}).get(cls, 0) < 30:
            fails.append(f"monitor_up={cls}: need >= 30 rows, have {c.get('monitor_up', {}).get(cls, 0)}")
    try:
        load_buttons(calibration_path)
    except (Uncalibrated, FileNotFoundError) as e:
        fails.append(f"calibration: {e}")
    return fails + foxy_problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("session")
    t.add_argument("--every", type=int, default=10)
    s = sub.add_parser("sheet")
    s.add_argument("session")
    s.add_argument("--cols", type=int, default=4)
    c = sub.add_parser("check")
    c.add_argument("session")
    c.add_argument("--ready", action="store_true")
    a = ap.parse_args(argv)
    d = Path(config.CORPUS_DIR) / a.session
    if a.cmd == "template":
        print("wrote", make_template(d, a.every))
    elif a.cmd == "sheet":
        print("wrote", len(make_sheets(d, a.cols)), "sheets in", d / "sheets")
    else:
        errs, rows = validate(d / "labels.csv", foxy_info(config.CORPUS_DIR)[0])
        print(json.dumps(counts(rows), indent=1))
        for e in errs:
            print("ERROR", e)
        if a.ready:
            fails = ready_failures(config.CORPUS_DIR, config.CALIBRATION_PATH)
            for f in fails:
                print("NOT READY:", f)
            if not fails:
                (Path(config.CORPUS_DIR) / "READY").write_text("ready\n")
                print("READY")
            errs += fails
        sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
