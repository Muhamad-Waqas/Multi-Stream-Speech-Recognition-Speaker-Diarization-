"""
Windows "Application Control policy has blocked this file" fix.

NeMo's mel-spectrogram preprocessor imports librosa, librosa imports numba, and numba's compiled file
(`_dispatcher.pyd`) is refused by Windows Application Control / Smart App Control:

    ImportError: DLL load failed while importing _dispatcher: An Application Control policy has blocked this file.

Our speech pipeline never needs numba's speed-ups (it only runs the models in PyTorch), so when numba cannot be
loaded we put a tiny do-nothing stand-in in its place BEFORE NeMo is imported. If numba loads fine, nothing changes.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from typing import Optional

_ROOTS = ("numba", "llvmlite")
STUB_REASON: Optional[str] = None       # the original import error, if the stand-in is active


class _Any:
    """Accepts any attribute, call or indexing. Used as a decorator it returns the function unchanged."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], _Any):
            return args[0]                           # @jit  /  @jit(...)(func)
        return self                                  # @jit(nopython=True)  -> acts as the decorator next

    def __getitem__(self, item):
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = sys.modules.get(f"{self.__name__}.{name}")
        if sub is not None:
            return sub
        if name.endswith("Warning"):
            value = type(name, (Warning,), {})
        elif name.endswith(("Error", "Exception")):
            value = type(name, (Exception,), {})
        else:
            value = _Any()
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []


_FINDER = _StubFinder()


def _make(name: str) -> _StubModule:
    mod = _StubModule(name)
    mod.__path__ = []
    mod.__spec__ = importlib.machinery.ModuleSpec(name, _FINDER, is_package=True)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, mod)
    return mod


def _install_stub() -> None:
    for name in list(sys.modules):                   # drop half-loaded real modules
        if name.split(".")[0] in _ROOTS:
            del sys.modules[name]
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)

    numba = _make("numba")
    numba.__version__ = "0.60.0"
    numba.jit = numba.njit = numba.vectorize = numba.guvectorize = numba.stencil = _Any()
    numba.generated_jit = numba.cfunc = _Any()
    numba.prange = range

    cuda = _make("numba.cuda")
    cuda.is_available = lambda: False                # NeMo then uses its plain-PyTorch code paths
    cuda.is_supported_version = lambda: False
    cuda.jit = _Any()
    for sub in ("numba.core", "numba.core.errors", "numba.types", "numba.typed", "numba.np", "numba.extending"):
        _make(sub)

    llvmlite = _make("llvmlite")
    llvmlite.__version__ = "0.43.0"


def is_stubbed() -> bool:
    return isinstance(sys.modules.get("numba"), _StubModule)


def fix_numba(force: bool = False) -> Optional[str]:
    """
    Call this BEFORE importing NeMo / torch-audio libraries.
    Returns None when numba works normally, or the original error text when the stand-in was installed.
    """
    global STUB_REASON
    if is_stubbed():
        return STUB_REASON
    reason = "forced"
    if not force:
        try:
            import numba  # noqa: F401
            return None
        except Exception as e:                       # blocked DLL, missing, or incompatible numba
            reason = f"{type(e).__name__}: {e}"
    _install_stub()
    STUB_REASON = reason
    return reason
