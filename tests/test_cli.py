import subprocess
import sys


def test_module_entrypoint_help():
    p = subprocess.run([sys.executable, "-m", "smo_sst", "--help"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "animate" in p.stdout and "benchmark" in p.stdout
