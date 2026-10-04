"""
Reproduces YOUR error: numba cannot be loaded because Windows Application Control blocks its _dispatcher file.
Each scenario runs in a fresh Python process so the fake modules cannot leak between tests.
"""
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
BLOCK_MSG = "DLL load failed while importing _dispatcher: An Application Control policy has blocked this file."

FAKE_LIBROSA = '''
import numba
from numba import jit, guvectorize, stencil, cuda
from numba.core.errors import NumbaWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaWarning)

@jit(nopython=True, cache=True)
def add_one(x):
    return x + 1

@numba.jit
def times_two(x):
    return x * 2

@guvectorize(["void(float32[:], float32[:])"], "(n)->(n)", nopython=True, cache=True)
def vec(x, y):
    y[:] = x

@stencil
def sten(a):
    return a[0]

def total(n):
    return sum(i for i in numba.prange(n))
'''


def run_py(code, modules):
    with tempfile.TemporaryDirectory() as d:
        for rel, text in modules.items():
            p = Path(d) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        r = subprocess.run([sys.executable, "-c", f"import sys; sys.path[:0]=[{d!r}, {ROOT!r}]\n" + textwrap.dedent(code)],
                           capture_output=True, text=True)
    return r


class NumbaBlockedTests(unittest.TestCase):
    BLOCKED = {"numba/__init__.py": f"raise ImportError({BLOCK_MSG!r})\n", "fakelibrosa.py": FAKE_LIBROSA}

    def test_without_fix_the_import_fails_like_your_error(self):
        r = run_py("import fakelibrosa", self.BLOCKED)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Application Control policy", r.stderr)

    def test_with_fix_librosa_style_code_imports_and_runs(self):
        r = run_py("""
            from mstt import compat
            reason = compat.fix_numba()
            assert reason and "Application Control" in reason, reason
            import fakelibrosa
            assert fakelibrosa.add_one(1) == 2 and fakelibrosa.times_two(4) == 8
            assert fakelibrosa.total(4) == 6
            import numba, importlib.util
            from numba import cuda
            assert cuda.is_available() is False            # NeMo must see "no numba CUDA" and use plain PyTorch
            assert importlib.util.find_spec("numba") is not None
            import numba.cuda.cudadrv.driver               # deep imports must not crash either
            from numba.core.errors import NumbaPerformanceWarning
            assert issubclass(NumbaPerformanceWarning, Warning)
            assert compat.is_stubbed()
            assert compat.fix_numba() == reason            # calling again (Streamlit reruns) is harmless
            print("OK")
        """, self.BLOCKED)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)

    def test_half_loaded_numba_is_cleaned_up(self):
        broken = {"numba/__init__.py": "import numba.partial\n", "numba/partial.py": f"raise ImportError({BLOCK_MSG!r})\n",
                  "fakelibrosa.py": FAKE_LIBROSA}
        r = run_py("""
            from mstt import compat
            assert compat.fix_numba()
            import fakelibrosa
            assert fakelibrosa.add_one(2) == 3
            print("OK")
        """, broken)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_working_numba_is_left_alone(self):
        good = {"numba/__init__.py": "__version__ = '9.9'\ndef jit(*a, **k):\n    return (lambda f: f)\n"}
        r = run_py("""
            from mstt import compat
            assert compat.fix_numba() is None
            import numba
            assert numba.__version__ == '9.9' and not compat.is_stubbed()
            print("OK")
        """, good)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_stub_attributes_never_loop_forever(self):
        r = run_py("""
            import inspect
            from mstt import compat
            compat.fix_numba(force=True)
            import numba
            inspect.unwrap(numba.jit)          # would hang if dunder lookups were answered
            inspect.signature(lambda: 0)
            import copy, pickle
            print("OK")
        """, {})
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
