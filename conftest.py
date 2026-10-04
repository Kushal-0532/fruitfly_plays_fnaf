from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    has_corpus = (Path(__file__).parent / "data/corpus/READY").exists()
    live_requested = "live" in (config.getoption("-m") or "")
    for it in items:
        if "corpus" in it.keywords and not has_corpus:
            it.add_marker(pytest.mark.skip(reason="data/corpus/READY missing"))
        if "live" in it.keywords and not live_requested:
            it.add_marker(pytest.mark.skip(reason="live test: run with -m live"))
