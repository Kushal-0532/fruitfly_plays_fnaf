"""New game, then let the fly play every night in turn, recording each round.
Click New Game once, then per night: Continue + main.py until the title menu's night label changes (= night won), max --tries per night.
Reuses scripts/auto_night.py pieces. Stop: touch logs/AUTO_STOP.

  python -m scripts.campaign --first-night 1 --last-night 5 -- --viz
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config
from actuator import Actuator
from capture import SCALE, Capture, find_window
from geometry import Geometry, Window, capture_to_ref
from scripts.auto_night import CLICK, CONT, NIGHT, TEMPLATE, death_report, grab, iou, mask, raise_game

NEW_GAME = (140, 226)  # capture px of the "New Game" text on the title menu


def menu(cont_tpl, timeout, stop):
    """Wait for the title menu -> its night-label mask (None on timeout/stop)."""
    t0 = time.time()
    while time.time() - t0 < timeout and not stop.exists():
        raise_game()
        cap = None
        try:
            cap = Capture()
            f = grab(cap)
        except RuntimeError as e:
            print("capture hiccup:", str(e)[:80], flush=True)
            time.sleep(3)
            continue
        finally:
            if cap:
                cap.close()
        if iou(mask(f, CONT), cont_tpl) >= 0.6:
            return mask(f, NIGHT)
        time.sleep(3)
    return None


def click(px):
    cap = Capture()
    try:
        grab(cap)
        wid, x, y, w, h = find_window()
        geom = Geometry(Window(wid, x, y, w, h), SCALE)
        rx, ry = capture_to_ref(geom, *px)
        Actuator(wid, geom, {"b": (rx - 4, ry - 4, 8, 8)}).click("b")
    finally:
        cap.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-night", type=int, default=1)
    ap.add_argument("--last-night", type=int, default=5)
    ap.add_argument("--tries", type=int, default=6, help="rounds per night before giving up")
    ap.add_argument("--no-new-game", action="store_true", help="Continue from the saved night instead of New Game")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    cont_tpl = np.load(TEMPLATE)["cont"]
    out, stop = Path(config.LOG_DIR) / "auto", Path(config.LOG_DIR) / "AUTO_STOP"
    out.mkdir(parents=True, exist_ok=True)
    night, tries, new_game = a.first_night, 0, not a.no_new_game
    while night <= a.last_night and tries < a.tries:
        label = menu(cont_tpl, 120, stop)
        if label is None:
            print("stopping:", "AUTO_STOP" if stop.exists() else "no title menu within timeout")
            return
        first = new_game
        if new_game:
            print("New Game -> Night 1", flush=True)
            click(NEW_GAME)
            new_game = False
        else:
            print(f"night {night} try {tries + 1}/{a.tries}: Continue", flush=True)
            click(CLICK)
        time.sleep(1.0)
        t0 = time.time()
        rec_name = f"c_n{night}_{int(t0)}"
        cmd = [sys.executable, "main.py", "--night", str(night), *extra, "--record", rec_name]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while proc.poll() is None:
            raise_game()
            time.sleep(4)
        o, e = proc.communicate()
        line = [l for l in o.splitlines() if l.startswith("RunSummary")]
        runs = sorted(Path(config.LOG_DIR).glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
        rec = {"night": night, "try": tries + 1, "secs": round(time.time() - t0), "summary": line[-1] if line else e[-300:],
               "run_log": str(runs[-1]) if runs else None, "corpus": rec_name}
        if runs:
            rec["death"] = death_report(runs[-1])
            try:
                from scripts.forensics import make_sheet
                rec["sheet"] = str(make_sheet(runs[-1], Path(config.CORPUS_DIR) / rec_name))
            except Exception as ex:  # forensics must never break the loop
                rec["sheet_error"] = repr(ex)[:200]
        with open(out / "campaign.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if stop.exists():
            return
        new = menu(cont_tpl, 120, stop)
        if new is None:
            print("stopping: no title menu after the round")
            return
        # won = the run reached 6 AM, or the menu's night label changed (not trusted right after New Game: the old label was a different save)
        if "status='6am'" in rec["summary"] or (not first and iou(new, label) < 0.6):
            print(f"NIGHT {night} WON", flush=True)
            night, tries = night + 1, 0
        else:
            tries += 1
    print("campaign over: night", night, "tries", tries)


if __name__ == "__main__":
    main()
