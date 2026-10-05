import os
import subprocess
import sys

import pytest

needs = pytest.mark.skipif(not (os.path.isdir("data/corpus/n2_a") and os.path.isdir("data/corpus/n1_a")
                                and os.path.exists("data/templates/fly_readout.npz")),
                           reason="needs the local corpus and the fitted fly readout")

# (session, first frame, last frame, max seconds until the door closes; None = nobody there, must not close[, side, default L])
CASES = [
    ("n2_a", 3640, 3700, 2.0),   # Bonnie appears at ~1.1 s
    ("n2_a", 2890, 2915, 1.5),
    ("n2_a", 3820, 3845, 2.0),
    ("n2_a", 5140, 5155, 2.0),
    ("n1_a", 4640, 4690, 2.5),   # Bonnie at ~1.6 s (unlabeled in labels.csv, visible in the frame)
    ("n2_a", 650, 690, None),    # lit empty hallways
    ("n2_a", 1465, 1500, None),
    # Chica in the lit right WINDOW (live 2026-10-05 N3 round 5, 418.7-421.6 s); the old doorway patch read 0.07 and she got in
    ("n3_readout_1791175509", 2038, 2062, 1.0, "R"),
    ("n3_readout_1791175509", 1970, 1990, None),  # lit empty window
]


@pytest.mark.slow
@needs
@pytest.mark.parametrize("sess,f0,f1,limit,side", [(*c, "L")[:5] for c in CASES])
def test_fly_in_the_loop(sess, f0, f1, limit, side):
    """Recorded frames through the real readers + fly + supervisor. Own process: flyvis cannot build a second network in a
    worker thread of a process that already built one (test isolation only; the live bot builds exactly one).
    Windows start cold (network seeded from the first frame), so a brightness step right at frame f0 can false-alarm; none of
    these windows do."""
    if not os.path.isdir(f"data/corpus/{sess}"):
        pytest.skip(f"no corpus {sess}")
    r = subprocess.run([sys.executable, "-m", "scripts.replay_run", f"data/corpus/{sess}", str(f0), str(f1)], capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": "."}, timeout=240)
    assert r.returncode == 0, r.stderr[-800:]
    closes = [l.split() for l in r.stdout.splitlines() if "threat_close" in l]
    if limit is None:
        assert not closes, "false close:\n" + r.stdout[-500:]
    else:
        assert closes and closes[0][2] == f"threat_close_{side}", "the fly never raised the alarm:\n" + r.stdout[-500:]
        assert float(closes[0][0]) < limit
