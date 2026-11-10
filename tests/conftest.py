import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))           # lets the tests run without installing


@pytest.fixture(scope="session")
def build(tmp_path_factory):
    """Return build(L, n_seeds, quick=True): compiles cpp/ising.cpp into a temporary
    folder and returns the path of the program. Skips if there is no g++."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    folder = tmp_path_factory.mktemp("bin")
    cache = {}

    def _build(L, n_seeds=2, seed_start=1, quick=True, source=None):
        # source: 'ising.cpp' (power of 2 only) or 'ising_any_L.cpp'; default = what the Makefile picks
        source = source or ("ising.cpp" if L & (L - 1) == 0 else "ising_any_L.cpp")
        key = (L, n_seeds, seed_start, quick, source)
        if key not in cache:
            exe = folder / f"ising_{L}_{n_seeds}_{seed_start}_{int(quick)}_{source[:-4]}"
            cmd = ["g++", "-O2", "-ffast-math", "-fopenmp", "-std=c++17",
                   f"-DL_SIZE={L}", f"-DN_SEEDS={n_seeds}", f"-DSEED_START={seed_start}"]
            if quick:
                cmd.append("-DQUICK")
            cmd += ["-o", str(exe), str(ROOT / "cpp" / source)]
            subprocess.run(cmd, check=True, capture_output=True)
            cache[key] = exe
        return cache[key]

    return _build


def run_program(exe, outdir, threads=1):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    subprocess.run([str(exe), str(outdir)], check=True, capture_output=True, env=env)


@pytest.fixture
def run():
    return run_program
