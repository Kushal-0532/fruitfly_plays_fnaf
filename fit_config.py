"""Phase 17: change tunables through config_fit.json["overrides"] with a printed diff and a y/n.
usage: python fit_config.py MIN_HOLD_S=4 CHECK_PERIOD='{"L":6,"R":6}'"""
import json
import sys
from pathlib import Path

import config

PATH = Path(config.POWER_FIT_PATH)


def main(pairs):
    d = json.loads(PATH.read_text()) if PATH.exists() else {}
    ov, new = dict(d.get("overrides", {})), {}
    for p in pairs:
        k, v = p.split("=", 1)
        if not hasattr(config, k):
            sys.exit(f"unknown config key {k}")
        new[k] = json.loads(v)
    for k, v in new.items():
        print(f"{k}: {ov.get(k, getattr(config, k))!r} -> {v!r}")
    if input("write? [y/N] ").strip().lower() == "y":
        d["overrides"] = {**ov, **new}
        PATH.write_text(json.dumps(d, indent=1))
        print("written")


if __name__ == "__main__":
    main(sys.argv[1:])
