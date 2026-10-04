import random

import pytest

from power import PowerModel, load_rates, save_rates

TRUE = {1: .05, 2: .1, 3: .15, 4: .2}


def feed(m, rates, levels=(1, 2, 3, 4), secs=60, seed=0):
    rnd, t = random.Random(seed), 0.0
    for u in levels:
        p = 100.0
        for i in range(secs * 10):
            m.observe(t, p - rates[u] * i / 10 + rnd.gauss(0, 0.1), u)
            t += 0.1


def test_fit_recovers_rates():
    m = PowerModel()
    feed(m, TRUE)
    got = m.fit()
    for u, r in TRUE.items():
        assert got[u] == pytest.approx(r, rel=0.05)


def test_fit_keeps_defaults():
    m = PowerModel()
    default = dict(m.rates)
    feed(m, TRUE, levels=(1, 2), secs=60)
    feed(m, TRUE, levels=(3,), secs=6)  # < 10 s after settle
    got = m.fit()
    assert got[3] == default[3] and got[4] == default[4] and got[1] != default[1]


def test_depletion():
    m = PowerModel()
    assert m.time_to_depletion(50, 2) == 50 / m.rate(2)
    assert m.rate(None) == max(m.rates.values())
    assert m.rate(9) == m.rate(4)


def test_conservation_boundary():
    m = PowerModel({1: .1, 2: .2, 3: .3, 4: .4})
    left = 3 * 89 * 1.15
    p = left * 0.2  # time_to_depletion == left exactly at usage 2
    assert not m.needs_conservation(p, 2, 3)
    assert m.needs_conservation(p - 0.01, 2, 3)


def test_mismatch():
    m = PowerModel({1: .1, 2: .2, 3: .3, 4: .4})
    for i in range(300):
        t = i * 0.1
        m.observe(t, 100 - 0.2 * t, 2)
    assert not m.mismatch(29.9, 100 - .2 * 29.9, 2)
    assert m.mismatch(29.9, 100 - .2 * 1.5 * 29.9, 2)
    m2 = PowerModel({1: .1, 2: .2, 3: .3, 4: .4})
    for i in range(80):
        m2.observe(i * 0.1, 100 - 0.2 * i * .1, 2)
    assert not m2.mismatch(7.9, 100 - .2 * 1.5 * 7.9, 2)  # < 10 s since segment start


def test_roundtrip(tmp_path):
    m = PowerModel()
    m.rates[2] = 0.123
    save_rates(m, tmp_path / "f.json")
    assert load_rates(tmp_path / "f.json").rates == m.rates
