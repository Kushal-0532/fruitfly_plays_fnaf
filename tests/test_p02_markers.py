import pytest


@pytest.mark.corpus
def test_corpus_marker_dummy():
    # skipped unless data/corpus/READY exists; if it ran, READY exists
    from pathlib import Path
    assert Path("data/corpus/READY").exists()


@pytest.mark.live
def test_live_marker_dummy():
    pytest.fail("live tests must be skipped by default")


def test_markers_skip_by_default(pytester=None):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_p02_markers.py", "-k", "dummy"],
                       capture_output=True, text=True)
    assert "2 skipped" in r.stdout or "skipped" in r.stdout, r.stdout
    assert "failed" not in r.stdout
