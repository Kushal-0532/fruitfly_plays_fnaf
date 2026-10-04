"""Hands-off retry loop: title menu -> click Continue -> run the bot -> bot exits (death / 6 AM / safe mode) -> back to the menu -> again.
Stops when the menu no longer says the expected night (= the night was won), after --rounds, or when logs/AUTO_STOP exists.
Per-round results go to logs/auto/rounds.jsonl. It never edits code: read the round's run log, fix, and the next round uses it.

  python -m scripts.auto_night --make-template --night 2     # game on the title menu: remembers what Continue / "Night 2" look like
  python -m scripts.auto_night --night 2 --rounds 5 -- --viz --record auto   # everything after -- goes to main.py
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
from geometry import Geometry, Window, capture_to_ref, load_buttons

TEMPLATE = "data/templates/menu.npz"
CONT = (slice(248, 274), slice(50, 200))   # ">> Continue" row (capture px)
NIGHT = (slice(274, 288), slice(85, 140))  # "Night N" under it
CLICK = (140, 262)                         # capture px of the Continue text
WHITE = 235


def mask(frame, box):
    return frame.astype(np.float32).mean(axis=2)[box] > WHITE


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def raise_game():
    """Game must be on top or the screencast captures whatever covers it (terminal, notifications)."""
    try:
        subprocess.run(["xdotool", "windowactivate", str(find_window()[0])], timeout=5, capture_output=True)
    except Exception:
        pass


def death_report(run_log):
    """Last trusted state + last actions of a run log, so the fixer (me) sees why it died without digging."""
    rows = [json.loads(l) for l in open(run_log)][1:]
    tr = [r for r in rows if r["trusted"]]
    last = tr[-1] if tr else {}
    acts = [(round(r["t"], 1), r["action"], r["reason"], r["hall"]) for r in rows if r["action"] != "NONE" and r["t"] >= last.get("t", 0) - 20]
    return {"last_t": last.get("t"), "hour": last.get("hour"), "power": last.get("power_pct"),
            "door": last.get("door_closed"), "cam": last.get("cam"), "last_actions": acts[-8:]}


def grab(cap):
    return np.asarray(cap.grab(timeout=20))


def make_template(night):
    cap = Capture()
    try:
        f = grab(cap)
    finally:
        cap.close()
    c = mask(f, CONT)
    assert c.sum() > 100, "the title menu (with >> Continue) is not on screen"
    np.savez(TEMPLATE, cont=c, **{f"night{night}": mask(f, NIGHT)})
    print(f"saved {TEMPLATE}: Continue row + night {night} label")


def menu_state(f, tpl, night):
    """-> (on_menu, is_expected_night)"""
    on = iou(mask(f, CONT), tpl["cont"]) >= 0.6
    key = f"night{night}"
    return on, bool(on and key in tpl.files and iou(mask(f, NIGHT), tpl[key]) >= 0.6)


def wait_menu(tpl, night, timeout, stop):
    t0 = time.time()
    while time.time() - t0 < timeout and not stop.exists():
        raise_game()
        cap = None
        try:
            cap = Capture()
            f = grab(cap)
        except RuntimeError as e:  # transient pipewire/gst hiccup: try again
            print("capture hiccup:", str(e)[:80], flush=True)
            time.sleep(3)
            continue
        finally:
            if cap:
                cap.close()
        on, same = menu_state(f, tpl, night)
        if on:
            return same
        time.sleep(3)
    return None


def click_continue():
    cap = Capture()
    try:
        grab(cap)
        wid, x, y, w, h = find_window()
        geom = Geometry(Window(wid, x, y, w, h), SCALE)
        rx, ry = capture_to_ref(geom, *CLICK)
        buttons = {"menu_continue": (rx - 4, ry - 4, 8, 8)}
        Actuator(wid, geom, buttons).click("menu_continue")
    finally:
        cap.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--menu-timeout", type=float, default=120)
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if a.make_template:
        return make_template(a.night)
    tpl = np.load(TEMPLATE)
    out, stop = Path(config.LOG_DIR) / "auto", Path(config.LOG_DIR) / "AUTO_STOP"
    out.mkdir(parents=True, exist_ok=True)
    for r in range(a.rounds):
        same = wait_menu(tpl, a.night, a.menu_timeout, stop)
        if stop.exists() or same is None:
            print("stopping:", "AUTO_STOP" if stop.exists() else "no title menu within timeout")
            return
        if not same:
            print(f"menu no longer says Night {a.night}: night won (or the menu changed). Stopping.")
            return
        print(f"round {r + 1}/{a.rounds}: Continue -> Night {a.night}", flush=True)
        click_continue()
        time.sleep(1.0)
        t0, cmd = time.time(), [sys.executable, "main.py", "--night", str(a.night), *extra]
        if "--record" in extra:  # one corpus folder per round
            cmd[cmd.index("--record") + 1] = f"{extra[extra.index('--record') + 1]}_{int(t0)}"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while proc.poll() is None:  # keep the game window on top for the whole night
            raise_game()
            time.sleep(4)
        res = subprocess.CompletedProcess(cmd, proc.returncode, *proc.communicate())
        line = [l for l in res.stdout.splitlines() if l.startswith("RunSummary")]
        runs = sorted(Path(config.LOG_DIR).glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
        rec = {"round": r + 1, "secs": round(time.time() - t0), "summary": line[-1] if line else res.stderr[-300:],
               "run_log": str(runs[-1]) if runs else None}
        if runs:
            rec["death"] = death_report(runs[-1])
            if "--record" in extra:
                try:
                    from scripts.forensics import make_sheet
                    rec["sheet"] = str(make_sheet(runs[-1], Path(config.CORPUS_DIR) / cmd[cmd.index("--record") + 1]))
                except Exception as e:  # forensics must never break the retry loop
                    rec["sheet_error"] = repr(e)[:200]
        with open(out / "rounds.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
