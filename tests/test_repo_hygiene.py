import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def test_setup_base_is_committed_executable():
    """Guarda contra commitar a mudança de modo 100755 -> 100644 em scripts/setup_base.sh."""
    out = subprocess.run(["git", "ls-files", "-s", "scripts/setup_base.sh"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout:
        pytest.skip("fora de um repositório git")
    assert out.stdout.split()[0] == "100755", out.stdout
