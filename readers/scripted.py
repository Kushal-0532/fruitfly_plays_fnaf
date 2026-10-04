"""Scripted reader for tests and sim: timeline = [(t_start, {field: value}), ...] applied piecewise
(each piece overrides the named fields from t_start on). A `world` (sim.World) supplies base values."""
from state import Readings


class ScriptedReader:
    def __init__(self, timeline, world=None):
        self.timeline, self.world = sorted(timeline, key=lambda x: x[0]), world

    def read(self, frame, t):
        d = dict(self.world.snapshot(t)) if self.world else {}
        for t0, fields in self.timeline:
            if t0 <= t:
                d.update(fields)
        return Readings(t, **d)
