"""Phase 28: contact sheet of the last seconds of a run, so a death can be classified by looking at one PNG.
  python -m scripts.forensics logs/run_X.jsonl data/corpus/auto_<ts>  -> logs/forensics/<run_ts>/{sheet.png,summary.json}"""
import bisect
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import config
from scripts.audit_decisions import audit

COLS, TW, N_AFTER = 4, 320, 6


def pick_frames(times, t_end, seconds, fps):
    """-> frame indices nearest to each target time in [t_end - seconds, t_end] (deduplicated, in order)."""
    out = []
    for k in range(int(seconds * fps) + 1):
        t = t_end - seconds + k / fps
        i = min(max(bisect.bisect_left(times, t), 0), len(times) - 1)
        if i > 0 and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        if not out or out[-1] != i:
            out.append(i)
    return out


def _f(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def _caption(r):
    d = r["door_closed"]
    b = lambda v: "-" if v is None else int(bool(v))
    l1 = (f"t={r['t']:.1f} h={r['hour']} p={_f(r['power_pct'], '{:.0f}')}% usage={r['usage']} mon={int(bool(r['monitor_up']))} "
          f"cam={r['cam'] or '-'} doors L{b(d['L'])} R{b(d['R'])} lights L{b(r['light_on']['L'])} R{b(r['light_on']['R'])}")
    return l1, f"hall L={_f(r['hall']['L'])} R={_f(r['hall']['R'])} | {r['action']} {r['reason']}"


def make_sheet(run_log, corpus_dir, out_dir=None, seconds=8.0, fps=2):
    rows = [r for r in map(json.loads, open(run_log)) if "header" not in r]
    corpus = Path(corpus_dir)
    times = json.load(open(corpus / "meta.json"))["times"]
    tr = [r for r in rows if r["trusted"]]
    last = tr[-1] if tr else rows[-1]
    t_end = last["t"]
    idx = pick_frames(times, t_end, seconds, fps)
    after = [i for i in range(bisect.bisect_right(times, t_end), len(times))][:N_AFTER]
    tiles = [(i, None) for i in idx] + [(i, f"after death +{times[i] - t_end:.1f} s") for i in after]
    row_t = [r["t"] for r in rows]
    shots = []
    for i, cap in tiles:
        im = Image.open(corpus / "frames" / f"{i:06d}.png").convert("RGB")
        im = im.resize((TW, round(TW * im.height / im.width)))
        tile = Image.new("RGB", (TW, im.height + 24))
        tile.paste(im, (0, 24))
        if cap:
            lines = (cap, "")
        else:
            k = min(max(bisect.bisect_left(row_t, times[i]), 0), len(rows) - 1)
            lines = _caption(rows[k])
        d = ImageDraw.Draw(tile)
        d.text((2, 1), lines[0][:60], fill=(255, 255, 0))
        d.text((2, 12), lines[1][:60], fill=(0, 255, 255))
        shots.append(tile)
    th = max(s.height for s in shots)
    nrow = -(-len(shots) // COLS)
    sheet = Image.new("RGB", (COLS * TW, nrow * th))
    for k, s in enumerate(shots):
        sheet.paste(s, ((k % COLS) * TW, (k // COLS) * th))
    out = Path(out_dir) if out_dir else Path(config.LOG_DIR) / "forensics" / Path(run_log).stem.removeprefix("run_")
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / "sheet.png")
    acts = [(round(r["t"], 1), r["action"], r["reason"]) for r in rows if r["action"] not in ("NONE", "SAFE_MODE")][-12:]
    summ = {"last_trusted": last, "last_actions": acts,
            "monitor_raised_last_10s": any(r["monitor_up"] for r in rows if t_end - 10 <= r["t"] <= t_end),
            "jam_events": [(round(r["t"], 1), r["reason"]) for r in rows if str(r["reason"]).startswith("jam_")],
            "door_audit": audit(rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return out / "sheet.png"


if __name__ == "__main__":
    print(make_sheet(sys.argv[1], sys.argv[2]))
