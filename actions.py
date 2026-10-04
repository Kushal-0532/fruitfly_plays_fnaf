"""Shared action vocabulary: policy emits Decisions, safety vets them, actuator performs them."""
from dataclasses import dataclass
from enum import Enum, auto


class Action(Enum):
    NONE = auto()
    DOOR_L = auto()
    DOOR_R = auto()
    LIGHT_L = auto()
    LIGHT_R = auto()
    MONITOR = auto()
    CAM = auto()
    SAFE_MODE = auto()


@dataclass
class Decision:
    action: Action
    arg: str | None = None  # cam id for Action.CAM, e.g. "1A"
    reason: str = ""
