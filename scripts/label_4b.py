"""labels_4b.csv from the agent's eyeballed overrides: everything 'empty' (freddy=0, chica=0) except the listed frames.
  python -m scripts.label_4b data/corpus/<s> 120:f 124:c 130:fc 140:x     (f = Freddy present, c = Chica present, x = unusable)"""
import csv
import sys
from collections import Counter
from pathlib import Path


def main(session, pairs):
    over = {int(a): b for a, b in (p.split(":") for p in pairs)}
    d = Path(session)
    idx = list(csv.DictReader(open(d / "cam_4b_index.csv")))
    n = Counter()
    with open(d / "labels_4b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "freddy", "chica", "look_id"])
        for r in idx:
            o = over.get(int(r["frame"]), "")
            fr, ch = ("x", "x") if o == "x" else (int("f" in o), int("c" in o))
            n[(fr, ch)] += 1
            w.writerow([r["frame"], fr, ch, r["look_id"]])
    return dict(n)


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2:]))
