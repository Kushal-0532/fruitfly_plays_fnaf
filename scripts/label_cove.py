"""Write labels_cove.csv for a session from the agent's eyeballed overrides: stage 1 everywhere except {frame: stage}.
  python -m scripts.label_cove data/corpus/<s> 491:2 493:3 ..."""
import csv
import sys
from collections import Counter
from pathlib import Path


def main(session, pairs):
    over = {int(a): int(b) for a, b in (p.split(":") for p in pairs)}
    d = Path(session)
    idx = list(csv.DictReader(open(d / "cove_index.csv")))
    with open(d / "labels_cove.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "stage", "look_id"])
        for r in idx:
            w.writerow([r["frame"], over.get(int(r["frame"]), 1), r["look_id"]])
    return Counter(over.get(int(r["frame"]), 1) for r in idx)


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2:]))
