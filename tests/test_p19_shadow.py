import numpy as np
import pytest

import ab_report
import shadow
from sim import simulate
from tests.fakes import night_bonnie_left_visit


def test_hallway_diff():
    a = np.full((20, 20), 7, np.uint8)
    assert shadow.hallway_diff(a, a).max() == 0
    b = a.copy()
    b[5:8, 5:8] = 90
    d = shadow.hallway_diff(b, a)
    assert d[5:8, 5:8].min() > 0 and d[:5].max() == 0


def test_compare_synthetic():
    y = [0] * 30 + [1] * 30
    rng = np.random.default_rng(0)
    sep = ab_report.compare(list(range(60)), list(range(60)), y)
    assert sep["pixel_auc"] == 1.0 and sep["promote"]
    noise = ab_report.compare(list(range(60)), rng.normal(size=60), y)
    assert noise["pixel_auc"] == 1.0 and not noise["promote"] and abs(noise["fly_auc"] - 0.5) < 0.2


def test_shadow_does_not_change_decisions():
    class Hook:
        def observe(self, frame, state, t):
            return {"fly_L": 1.0, "fly_R": None}
    _, plain, _ = simulate(night_bonnie_left_visit(), 60)
    _, shad, rows = simulate(night_bonnie_left_visit(), 60, shadow=Hook())
    assert [(t, d.action, d.arg) for t, d in plain] == [(t, d.action, d.arg) for t, d in shad]
    assert rows[-1]["fly_L"] == 1.0


@pytest.mark.slow
def test_fly_scorer_direction_and_no_carry_over():
    import perception
    sc = shadow.FlyScorer()
    bar = list(perception.moving_bar(1, n=shadow.N_BURST))
    static = [np.full((200, 200), 60, np.uint8)] * shadow.N_BURST
    moving, still = sum(sc.score_burst(bar).values()), sum(sc.score_burst(static).values())
    assert moving > still
    again = sum(sc.score_burst(bar).values())
    assert abs(moving - again) < 1e-6
