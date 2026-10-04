import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_is_311():
    assert sys.version_info[:2] == (3, 11)


@pytest.mark.parametrize("mod", ["flyvis", "torch", "PIL", "dbus_next", "numpy", "scipy", "sklearn"])
def test_imports(mod):
    importlib.import_module(mod)


def test_tools_present():
    for cmd in (["xdotool", "--version"], ["pw-record", "--version"], ["/usr/bin/python3", "-c", "import gi"]):
        assert subprocess.run(cmd, capture_output=True).returncode == 0, cmd


def test_weights_present():
    assert (ROOT / "data/results/flow/0000/000").exists()


def test_env_file():
    assert (ROOT / ".env").read_text().strip() == f"FLYVIS_ROOT_DIR={ROOT / 'data'}"


def test_freeze_backup():
    assert (ROOT / "logs/venv_py314_freeze.txt").exists()


def test_flyvis_loads():
    os.environ["FLYVIS_ROOT_DIR"] = str(ROOT / "data")
    from flyvis import NetworkView

    net = NetworkView("flow/0000/000").init_network()
    assert len(net.connectome.nodes.type[:]) > 40000
