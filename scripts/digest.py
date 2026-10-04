"""One-screen summary of a run log: what the bot did and what the fly saw. `python -m scripts.digest logs/run_*.jsonl`"""
import json
import sys


def digest(path):
    rows = [r for r in map(json.loads, open(path)) if "header" not in r]
    tr = [r for r in rows if r["trusted"]] or rows
    n = len(tr)
    mx = lambda k: max((r[k] for r in tr if r[k] is not None), default=None)
    hall = lambda s: max((r["hall"][s] for r in tr if r["hall"][s] is not None), default=None)
    acts = [(round(r["t"]), r["action"], r["reason"]) for r in rows if r["action"] != "NONE"]
    return {"log": str(path), "secs": round(rows[-1]["t"]), "last_hour": next((r["hour"] for r in reversed(tr) if r["hour"] is not None), None),
            "power_curve": [(round(r["t"]), r["power_pct"]) for r in tr[:: max(1, n // 8)] if r["power_pct"] is not None],
            "door_closed_frac": {s: round(sum(bool(r["door_closed"][s]) for r in tr) / n, 2) for s in "LR"},
            "monitor_up_frac": round(sum(bool(r["monitor_up"]) for r in tr) / n, 2),
            "peak": {"hall_L": hall("L"), "hall_R": hall("R"), "cove": mx("cove"), "cam4b": mx("cam4b")},
            "guards": sorted({r["guard"] for r in rows if r["guard"]}), "door_clicks": [a for a in acts if a[1].startswith("DOOR")], "actions": acts}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(digest(p), indent=1))
