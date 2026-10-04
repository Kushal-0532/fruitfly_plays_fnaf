"""SPEC D12: every door click must carry a fly reason code. Usage: python -m scripts.audit_decisions logs/run_X.jsonl [...]"""
import json
import sys

from policy import FLY_DOOR_REASONS

DOORS = ("DOOR_L", "DOOR_R")


def audit(rows):
    n, by, bad, guards = 0, {}, [], 0
    for r in rows:
        if r.get("guard"):
            guards += 1
        a = r.get("action")
        a = getattr(a, "name", a)
        if a not in DOORS:
            continue
        n += 1
        by[r["reason"]] = by.get(r["reason"], 0) + 1
        if r["reason"] not in FLY_DOOR_REASONS:
            bad.append((r.get("t"), a, r["reason"]))
    return {"door_actions": n, "by_reason": by, "violations": bad, "guard_overrides": guards}


def main(paths):
    code = 0
    for p in paths:
        rows = [j for j in map(json.loads, open(p)) if "header" not in j]
        res = audit(rows)
        print(p, {k: v for k, v in res.items() if k != "violations"})
        for v in res["violations"]:
            print("  VIOLATION", v)
        code |= bool(res["violations"])
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
