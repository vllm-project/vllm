# Minimal stub of tblib used for test environment without dependency.
# Provides a no-op install() emulating tblib.pickling_support.install.

class _PicklingSupport:
    def install(self) -> None:  # pragma: no cover - simple stub
        return None

pickling_support = _PicklingSupport()

__all__ = ["pickling_support"]
