"""Phase 06: power drain model. Pure logic; rates are placeholders until fitted on live logs (Phase 17)."""
import json

import config

SETTLE_S = 2.0  # drop samples this long after a usage change (bars lag the real state)


class PowerModel:
    def __init__(self, rates=None):
        self.rates = dict(rates or config.POWER_RATE_PCT_PER_S)
        self.segs = []     # list of (usage, [(t, p), ...]) contiguous same-usage runs
        self._change_t = None
        self._last_usage = None

    def rate(self, usage):
        if usage is None:
            return max(self.rates.values())
        return self.rates[min(max(int(usage), 1), 4)]

    def time_to_depletion(self, power_pct, usage):
        r = self.rate(usage)
        return float("inf") if r <= 0 else power_pct / r

    def seconds_left(self, hour):
        return (6 - (0 if hour is None else hour)) * config.HOUR_S

    def needs_conservation(self, power_pct, usage, hour, margin=1.15):
        return self.time_to_depletion(power_pct, usage) < self.seconds_left(hour) * margin

    def observe(self, t, power_pct, usage):
        if power_pct is None or usage is None:
            return
        if usage != self._last_usage:
            self._last_usage, self._change_t = usage, t
            self.segs.append((usage, []))
        if t - self._change_t < SETTLE_S:
            return
        self.segs[-1][1].append((t, power_pct))

    def fit(self):
        for level in (1, 2, 3, 4):
            n = span = sxy = sxx = 0
            for u, pts in self.segs:
                if u != level or len(pts) < 2:
                    continue
                mt = sum(p[0] for p in pts) / len(pts)
                mp = sum(p[1] for p in pts) / len(pts)
                sxy += sum((a - mt) * (b - mp) for a, b in pts)
                sxx += sum((a - mt) ** 2 for a, _ in pts)
                n += len(pts)
                span += pts[-1][0] - pts[0][0]
            if n >= 5 and span >= 10 and sxx > 0:
                self.rates[level] = max(-sxy / sxx, 0.0)
        return dict(self.rates)

    def mismatch(self, t, power_pct, usage, tol=0.20):
        if power_pct is None or usage is None or not self.segs:
            return False
        u, pts = self.segs[-1]
        if u != usage or not pts or t - pts[0][0] < 10:
            return False
        pred = self.rates[min(max(int(usage), 1), 4)] * (t - pts[0][0])
        actual = pts[0][1] - power_pct
        # power reads in whole percents, so allow +-1.5 points of quantisation noise on top of the relative tolerance
        return pred > 0 and abs(actual - pred) > max(tol * pred, 1.5)


def save_rates(model, path=config.POWER_FIT_PATH):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        d = {}
    d["power_rates"] = {str(k): v for k, v in model.rates.items()}
    json.dump(d, open(path, "w"), indent=1)


def load_rates(path=config.POWER_FIT_PATH):
    return PowerModel({int(k): v for k, v in json.load(open(path))["power_rates"].items()})


def fit_from_logs(paths):
    """Fit drain rates from run logs (jsonl rows with t, power_pct, usage); each file is its own timeline."""
    m = PowerModel()
    for p in paths:
        m._last_usage, m._change_t = None, 0.0
        for line in open(p):
            r = json.loads(line)
            if "header" not in r:
                m.observe(r["t"], r.get("power_pct"), r.get("usage"))
    m.fit()
    return m


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] != ["fit"] or len(sys.argv) < 3:
        sys.exit("usage: python -m power fit logs/run_*.jsonl")
    model = fit_from_logs(sys.argv[2:])
    save_rates(model)
    print("fitted rates (% per s):", model.rates)
