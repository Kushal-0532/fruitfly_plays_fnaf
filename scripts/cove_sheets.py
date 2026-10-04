"""Phase 32: contact sheets of cam-1C looks so the agent can label Foxy's stage by eye.
  python -m scripts.cove_sheets data/corpus/<session> logs/run_X.jsonl   -> <session>/cove_sheets/sheet_NN.png + cove_index.csv"""
import bisect
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import config
from fly_brain import CAM_VIEW

COLS, ROWS, TW = 6, 4, 240
READ_S = 1.0      # s after the camera switch before frames count (cut transient)
STEP, MAX_PER_LOOK = 3, 6


def looks(rows, cam="1C"):
    """-> list of lists of rows: consecutive monitor_up rows with the given cam."""
    out, cur = [], []
    for r in rows:
        if r["monitor_up"] and r["cam"] == cam:
            cur.append(r)
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def pick(times, look):
    """Frame indices of one look: from READ_S after it started, every STEP-th, at most MAX_PER_LOOK."""
    t0, t1 = look[0]["t"] + READ_S, look[-1]["t"]
    i0, i1 = bisect.bisect_left(times, t0), bisect.bisect_right(times, t1)
    return list(range(i0, i1, STEP))[:MAX_PER_LOOK]


def make(session, run_log, cam="1C"):
    sd = Path(session)
    rows = [r for r in map(json.loads, open(run_log)) if "header" not in r]
    times = json.load(open(sd / "meta.json"))["times"]
    tag = "cove" if cam == "1C" else f"cam_{cam.lower()}"
    out = sd / f"{tag}_sheets"
    out.mkdir(exist_ok=True)
    items = [(i, k, times[i]) for k, lk in enumerate(looks(rows, cam)) for i in pick(times, lk)]
    per = COLS * ROWS
    index = []
    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        tiles = []
        for n, (i, k, t) in enumerate(chunk):
            im = Image.open(sd / "frames" / f"{i:06d}.png").convert("RGB")
            w, h = im.size
            x0, x1, y0, y1 = CAM_VIEW
            im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            im = im.resize((TW, round(TW * im.height / im.width)))
            ImageDraw.Draw(im).text((3, 3), f"{i}", fill=(255, 255, 0))
            tiles.append(im)
            index.append((f"sheet_{s // per:02d}.png", n, i, round(t, 2), k))
        th = tiles[0].height
        sheet = Image.new("RGB", (COLS * TW, -(-len(tiles) // COLS) * th))
        for n, im in enumerate(tiles):
            sheet.paste(im, ((n % COLS) * TW, (n // COLS) * th))
        sheet.save(out / f"sheet_{s // per:02d}.png")
    with open(sd / f"{tag}_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "tile", "frame", "t", "look_id"])
        w.writerows(index)
    return len(items)


if __name__ == "__main__":
    cam = sys.argv[sys.argv.index("--cam") + 1] if "--cam" in sys.argv else "1C"
    print(make(sys.argv[1], sys.argv[2], cam), "tiles")
