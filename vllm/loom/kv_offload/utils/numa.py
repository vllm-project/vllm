from __future__ import annotations

from contextlib import contextmanager
import ctypes


def _load_libnuma() -> ctypes.CDLL | None:
    for name in ("libnuma.so.1", "libnuma.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


@contextmanager
def numa_membind(node: int | None, *, strict: bool = False):
    if node is None:
        yield
        return

    lib = _load_libnuma()
    if lib is None:
        if strict:
            raise RuntimeError(
                "NUMA membind requested but libnuma is not available (libnuma.so)."
            )
        yield
        return

    try:
        lib.numa_available.restype = ctypes.c_int
        if lib.numa_available() == -1:
            if strict:
                raise RuntimeError("NUMA membind requested but NUMA is not available")
            yield
            return

        free_fn = None
        if hasattr(lib, "numa_free_nodemask"):
            free_fn = lib.numa_free_nodemask
        elif hasattr(lib, "numa_bitmask_free"):
            free_fn = lib.numa_bitmask_free
        if free_fn is None:
            if strict:
                raise RuntimeError(
                    "NUMA membind requested but libnuma is missing a free function "
                    "(expected numa_free_nodemask or numa_bitmask_free)."
                )
            yield
            return

        free_fn.argtypes = [ctypes.c_void_p]
        free_fn.restype = None

        lib.numa_parse_nodestring.restype = ctypes.c_void_p
        lib.numa_parse_nodestring.argtypes = [ctypes.c_char_p]

        lib.numa_get_mems_allowed.restype = ctypes.c_void_p
        lib.numa_set_membind.argtypes = [ctypes.c_void_p]
        lib.numa_set_strict.argtypes = [ctypes.c_int]

        prev = lib.numa_get_mems_allowed()
        mask = lib.numa_parse_nodestring(str(int(node)).encode("utf-8"))
        if not mask:
            if strict:
                raise RuntimeError(f"NUMA membind requested but failed to parse node: {node}")
        else:
            lib.numa_set_membind(mask)
            lib.numa_set_strict(1)

        try:
            yield
        finally:
            if prev:
                lib.numa_set_membind(prev)
                lib.numa_set_strict(0)
            if mask:
                free_fn(mask)
            if prev:
                free_fn(prev)
    except Exception:
        if strict:
            raise
        yield
