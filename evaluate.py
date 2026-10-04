"""Phase 16: run logs (Phase 09 jsonl) -> success metrics with Wilson 95% intervals."""
import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Run:
    header: dict
    rows: list


@dataclass
class RunStats:
    night: int | None
    reached_6am: bool
    last_hour: int | None
    death_cause: str          # 6am | power_out | safe_mode | jumpscare_or_unknown
    power_at_hour5: float | None
    frac_monitor_up: float    # fraction of rows with monitor_up True
    frac_door_closed: float   # mean over rows and both sides of door_closed True
    action_counts_by_reason: dict
    safe_reason: str | None


def load_run(path):
    lines = [json.loads(l) for l in open(path) if l.strip()]
    return Run(lines[0]["header"], lines[1:])


def summarize(run):
    rows = run.rows
    hours = [r["hour"] for r in rows if r.get("hour") is not None]
    reached = 6 in hours
    last = rows[-1] if rows else {}
    safe = (last.get("guard") or "").startswith("SAFE_MODE")
    safe_reason = last["guard"].split(":", 1)[1] if safe else None
    powers = [r["power_pct"] for r in rows if r.get("trusted") and r.get("power_pct") is not None]
    p5 = next((r["power_pct"] for r in rows if r.get("hour") == 5 and r.get("power_pct") is not None), None)
    sides = [v for r in rows for v in (r.get("door_closed") or {}).values()]
    cause = ("6am" if reached else "power_out" if powers and powers[-1] <= 1
             else "safe_mode" if safe else "jumpscare_or_unknown")
    n = max(len(rows), 1)
    return RunStats(
        night=run.header.get("night"), reached_6am=reached, last_hour=hours[-1] if hours else None,
        death_cause=cause, power_at_hour5=p5,
        frac_monitor_up=sum(1 for r in rows if r.get("monitor_up")) / n,
        frac_door_closed=sum(1 for v in sides if v) / max(len(sides), 1),
        action_counts_by_reason=dict(Counter(r["reason"] for r in rows if r.get("action") != "NONE")),
        safe_reason=safe_reason)


def wilson(k, n, z=1.96):
    if n <= 0:
        raise ValueError("n must be > 0")
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compare(runs_a, runs_b):
    """runs_*: lists of RunStats."""
    out = {}
    for k, runs in (("a", runs_a), ("b", runs_b)):
        wins = sum(r.reached_6am for r in runs)
        out[f"rate_{k}"] = wins / len(runs)
        out[f"ci_{k}"] = wilson(wins, len(runs))
    (alo, ahi), (blo, bhi) = out["ci_a"], out["ci_b"]
    out["overlap"] = alo <= bhi and blo <= ahi
    return out


def rate_line(label, stats):
    k, n = sum(s.reached_6am for s in stats), len(stats)
    lo, hi = wilson(k, n)
    return f"{label}: {k}/{n} reached 6 AM ({100 * k / n:.1f}%, 95% CI {100 * lo:.1f}%-{100 * hi:.1f}%)"


def load_folder(folder, night=None):
    stats = [summarize(load_run(p)) for p in sorted(Path(folder).glob("run_*.jsonl"))]
    return [s for s in stats if night is None or s.night == night]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--night", type=int)
    ap.add_argument("--vs")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    stats = load_folder(a.folder, a.night)
    if not stats:
        raise SystemExit("no runs found")
    for i, s in enumerate(stats):
        print(f"run {i}: hour {s.last_hour} {s.death_cause} power@5AM={s.power_at_hour5} "
              f"monitor_up={s.frac_monitor_up:.2f} doors_closed={s.frac_door_closed:.2f}")
    print(rate_line(f"Night {a.night}" if a.night else "All nights", stats))
    result = {"runs": [asdict(s) for s in stats]}
    if a.vs:
        other = load_folder(a.vs, a.night)
        result["compare"] = c = compare(stats, other)
        print(rate_line("vs", other), "| intervals", "overlap" if c["overlap"] else "do not overlap")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
