# ABOUTME: Minimal stub exposing tblib pickling helpers for tests.
# ABOUTME: Provides a no-op install() matching tblib.pickling_support.install.

class _PicklingSupport:
    def install(self) -> None:  # pragma: no cover - simple stub
        return None

pickling_support = _PicklingSupport()

__all__ = ["pickling_support"]
