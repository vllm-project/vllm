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
def numa_membind(node: int | None):
    if node is None:
        yield
        return

    lib = _load_libnuma()
    if lib is None:
        yield
        return

    try:
        lib.numa_available.restype = ctypes.c_int
        if lib.numa_available() == -1:
            yield
            return

        lib.numa_parse_nodestring.restype = ctypes.c_void_p
        lib.numa_parse_nodestring.argtypes = [ctypes.c_char_p]

        lib.numa_get_mems_allowed.restype = ctypes.c_void_p
        lib.numa_set_membind.argtypes = [ctypes.c_void_p]
        lib.numa_set_strict.argtypes = [ctypes.c_int]
        lib.numa_free_nodemask.argtypes = [ctypes.c_void_p]

        prev = lib.numa_get_mems_allowed()
        mask = lib.numa_parse_nodestring(str(int(node)).encode("utf-8"))
        if mask:
            lib.numa_set_membind(mask)
            lib.numa_set_strict(1)

        try:
            yield
        finally:
            if prev:
                lib.numa_set_membind(prev)
                lib.numa_set_strict(0)
            if mask:
                lib.numa_free_nodemask(mask)
            if prev:
                lib.numa_free_nodemask(prev)
    except Exception:
        yield
