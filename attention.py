"""Where to look next. Each hallway light has a drive that climbs with time since the last look; an animatronic the fly saw there
makes it climb faster. The fly's evidence, not a fixed rota, decides which side wins. (Camera looks are the stall flips in policy.py.)
Pure and deterministic given the calls. All numbers U until fitted live."""
import math

import config


class Attention:
    def __init__(self, cfg=config):
        self.c = cfg
        self.last = {}                 # side -> time of the last look
        self.heat = {}                 # side -> (value, time)

    def _heat(self, k, t):
        v, t0 = self.heat.get(k, (0.0, t))
        return v * math.exp(-(t - t0) / self.c.ATTN_HEAT_TAU_S)

    def _add_heat(self, k, amount, t):
        self.heat[k] = (min(self._heat(k, t) + amount, self.c.ATTN_HEAT_MAX), t)

    def saw_hall(self, side, danger, t):
        if danger is not None and danger >= self.c.HALL_THRESH:
            self._add_heat(side, 1.0, t)

    def score(self, k, t, scale):
        return (t - self.last.get(k, 0.0)) / (self.c.CHECK_PERIOD[k] * scale) * (1.0 + self.c.ATTN_HEAT_GAIN * self._heat(k, t))

    def choose(self, t, hall_ok, scale, mult=1.0):
        """-> (side, score) with the highest score >= 1, or None. hall_ok: sides allowed."""
        best = None
        for k in hall_ok:
            sc = self.score(k, t, scale * mult)
            if sc >= 1.0 and (best is None or sc > best[1]):
                best = (k, sc)
        return best
