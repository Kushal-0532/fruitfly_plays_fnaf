"""Readers turn a frame into (partial) Readings. `Readers` merges several of them."""
import dataclasses
from typing import Protocol

from state import DICT_FIELDS, Readings


class Reader(Protocol):
    def read(self, frame, t: float) -> Readings: ...


def merge(a: Readings, b: Readings) -> Readings:
    """b wins over a, but a None in b never overwrites a value from a."""
    out = dataclasses.replace(a)
    for f in dataclasses.fields(Readings):
        if f.name == "t":
            continue
        bv, av = getattr(b, f.name), getattr(a, f.name)
        if f.name in DICT_FIELDS:
            m = dict(av or {})
            m.update({k: v for k, v in (bv or {}).items() if v is not None})
            setattr(out, f.name, m or None)
        elif bv is not None:
            setattr(out, f.name, bv)
    return out


class Readers:
    def __init__(self, readers):
        self.readers = list(readers)

    def read(self, frame, t):
        out = Readings(t)
        for r in self.readers:
            out = merge(out, r.read(frame, t))
        return out



class LiveReader:
    """Digits + buttons + hallway. The hallway score needs to know the light is on, which the button reader supplies."""

    def __init__(self, digits, buttons, hallway):
        self.digits, self.buttons, self.hallway = digits, buttons, hallway

    def read(self, frame, t):
        r = merge(self.digits.read(frame, t), self.buttons.read(frame, t))
        if r.monitor_up is False:  # the hallway ROIs are only the office while the monitor is down
            r = merge(r, self.hallway.read(frame, t, light_on=r.light_on))
        elif r.monitor_up and hasattr(self.hallway, "cam_read"):  # camera view: the fly watches it too
            r = merge(r, self.hallway.cam_read(frame, t, r.cam))
        return r


def live_reader(geom, buttons, hallway=None):
    """hallway: any object with read(frame, t, light_on=) -> Readings(hall=...). Default is the pixel-difference detector."""
    from readers.buttons import ButtonReader
    from readers.digits import DigitReader
    from readers.hallway import HallwayDetector
    return LiveReader(DigitReader(geom, buttons), ButtonReader(geom, buttons), hallway or HallwayDetector(geom, buttons))
