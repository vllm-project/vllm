"""ROCm Windows distributed stubs — meta-path import hook + ops patch.

Patches missing torch.distributed C++ backend modules and operator namespaces
that aren't compiled in the ROCm Windows PyTorch build.

Uses:
  1. Pre-injection of sys.modules for C-extension modules
  2. Meta-path import hook to intercept distributed/torchaudio imports
  3. _OpNamespace.__getattr__ monkey-patch for c10d/dtensor operator lookups
  4. Persistent BlackHoleOp cache for operator registration

Adapted from the Unsloth Studio Windows ROCm compatibility layer pattern.
"""

import sys
from types import ModuleType
from importlib.machinery import ModuleSpec


class AuditableMock(ModuleType):
    """Dynamic mock that generates any attribute on demand via __getattr__."""

    __file__ = "rocm_dist_fixes.py"
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            return ""
        if name[0].isupper():
            cls = type(name, (), {
                "__module__": self.__name__,
                "__name__": name,
                "__file__": "rocm_dist_fixes.py",
            })
            setattr(self, name, cls)
            return cls
        sub = AuditableMock(f"{self.__name__}.{name}")
        setattr(self, name, sub)
        return sub

    def __call__(self, *args, **kwargs):
        return AuditableMock("rocm_dummy_instance")

    def endswith(self, *args, **kwargs):
        return False

    def __iter__(self):
        return iter([])


_MOCK_OPS_CACHE: dict = {}


def _inject_dist_missing_names():
    """Inject names into torch.distributed that are missing on ROCm Windows."""
    try:
        import torch.distributed as dist

        if not hasattr(dist, "Backend"):
            dist.Backend = type("Backend", (), {
                "UNDEFINED": "undefined", "NCCL": "nccl",
                "GLOO": "gloo", "MPI": "mpi",
            })
        if not hasattr(dist, "Store"):
            dist.Store = type("Store", (), {})
        if not hasattr(dist, "PrefixStore"):
            dist.PrefixStore = type("PrefixStore", (), {})
        if not hasattr(dist, "TCPStore"):
            dist.TCPStore = type("TCPStore", (), {})
        if not hasattr(dist, "FileStore"):
            dist.FileStore = type("FileStore", (), {})
        if not hasattr(dist, "ReduceOp"):
            dist.ReduceOp = type("ReduceOp", (), {
                "SUM": 0, "AVG": 1, "PRODUCT": 2, "MIN": 3, "MAX": 4,
            })
    except Exception:
        pass


def _patch_torch_ops():
    """Patch torch._ops._OpNamespace.__getattr__ for distributed/DTensor namespaces."""
    try:
        import torch

        if hasattr(torch, "_ops") and hasattr(torch._ops, "_OpNamespace"):
            orig_getattr = torch._ops._OpNamespace.__getattr__

            def safe_getattr(self, name):
                if name == "_name":
                    return orig_getattr(self, name)

                ns_name = None
                try:
                    ns_name = self._name
                except Exception:
                    pass

                if not ns_name:
                    try:
                        for k in dir(torch.ops):
                            if getattr(torch.ops, k, None) is self:
                                ns_name = k
                                break
                    except Exception:
                        pass

                # Intercept missing distributed and DTensor namespaces
                if ns_name in (
                    "c10d", "c10d_functional", "_c10d_functional",
                    "dtensor", "_dtensor",
                ):
                    cache_key = f"{ns_name}.{name}"
                    if cache_key not in _MOCK_OPS_CACHE:

                        class BlackHoleOp:
                            def __init__(self, path):
                                self._path = path

                            def __getattr__(self, attr):
                                sub = f"{self._path}.{attr}"
                                if sub not in _MOCK_OPS_CACHE:
                                    _MOCK_OPS_CACHE[sub] = BlackHoleOp(sub)
                                return _MOCK_OPS_CACHE[sub]

                            def __call__(self, *args, **kwargs):
                                return self

                            def __iter__(self):
                                return iter([])

                            def __hash__(self):
                                return hash(self._path)

                            def __eq__(self, other):
                                return isinstance(other, BlackHoleOp) and self._path == other._path

                        _MOCK_OPS_CACHE[cache_key] = BlackHoleOp(cache_key)
                    return _MOCK_OPS_CACHE[cache_key]

                return orig_getattr(self, name)

            torch._ops._OpNamespace.__getattr__ = safe_getattr
    except Exception:
        pass


class ExecHookWrapper:
    """Wraps a loader to run post-exec hooks after the module loads."""

    def __init__(self, original_loader):
        self._original_loader = original_loader

    def __getattr__(self, name):
        return getattr(self._original_loader, name)

    def exec_module(self, module):
        self._original_loader.exec_module(module)
        if module.__name__ == "torch":
            _patch_torch_ops()
        elif module.__name__ == "torch.distributed":
            _inject_dist_missing_names()


# Intercepted modules (C-extensions not compiled on ROCm Windows, plus torchaudio)
_INTERCEPT_MODULES = {
    "torch._C._distributed_c10d",
    "torch._C._distributed_rpc",
    "torch.distributed._functional_collectives",
    "torch.distributed._symmetric_memory",
    "torchaudio",
}

_INTERCEPT_PREFIXES = (
    "torch._C._distributed_c10d.",
    "torch._C._distributed_rpc.",
    "torch.distributed._functional_collectives.",
    "torch.distributed._symmetric_memory.",
    "torchaudio.",
)


class RocmDistImportHook:
    """Meta-path import hook that mocks missing distributed modules."""

    def _should_intercept(self, fullname: str) -> bool:
        if fullname in _INTERCEPT_MODULES:
            return True
        for prefix in _INTERCEPT_PREFIXES:
            if fullname.startswith(prefix):
                return True
        return False

    def find_spec(self, fullname, path, target=None):
        # Wrap torch's loader to apply ops patch after it loads
        if fullname in ("torch", "torch.distributed"):
            sys.meta_path.remove(self)
            try:
                from importlib.util import find_spec
                real_spec = find_spec(fullname)
                if real_spec is not None and real_spec.loader is not None:
                    real_spec.loader = ExecHookWrapper(real_spec.loader)
                    return real_spec
            finally:
                sys.meta_path.insert(0, self)

        if self._should_intercept(fullname):
            class MockLoader:
                def create_module(self, spec):
                    m = AuditableMock(spec.name)
                    if spec.name == "torch.distributed":
                        m.is_available = lambda: False
                    return m

                def exec_module(self, module):
                    pass

            return ModuleSpec(fullname, MockLoader())

        return None


def apply_dist_fixes():
    """Install all ROCm Windows compatibility fixes."""
    # Clean up any existing instance of our hook
    sys.meta_path = [
        h for h in sys.meta_path
        if not isinstance(h, RocmDistImportHook)
    ]
    sys.meta_path.insert(0, RocmDistImportHook())

    # Pre-inject mocks for C-extension modules
    if "torch._C._distributed_c10d" not in sys.modules:
        sys.modules["torch._C._distributed_c10d"] = AuditableMock(
            "torch._C._distributed_c10d"
        )
    if "torch._C._distributed_rpc" not in sys.modules:
        sys.modules["torch._C._distributed_rpc"] = AuditableMock(
            "torch._C._distributed_rpc"
        )
    if "torch._C._distributed_rpc" in sys.modules:
        mock_rpc = sys.modules["torch._C._distributed_rpc"]
        if not hasattr(mock_rpc, "PyRRef"):
            mock_rpc.PyRRef = mock_rpc._PyRRef

    # If torch is already imported, patch ops directly
    if "torch" in sys.modules:
        _patch_torch_ops()

    # Inject missing names into torch.distributed
    if "torch.distributed" in sys.modules:
        d = sys.modules["torch.distributed"]
        missing_names = {
            "PrefixStore": None,
            "Store": None,
            "TCPStore": None,
            "FileStore": None,
            "Backend": None,
            "ReduceOp": None,
        }
        for name in missing_names:
            if not hasattr(d, name):
                setattr(d, name, AuditableMock(f"torch.distributed.{name}"))
