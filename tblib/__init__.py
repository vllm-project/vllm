"""A minimal stub of the 'tblib' package used in the test suite.

This stub provides the subset of the API required for the project's tests to run
without the actual external dependency. It defines a dummy `pickling_support`
module with an `install` function that does nothing, mirroring the real package's
behaviour for the purposes of import side‑effects.
"""

class _PicklingSupport:
    @staticmethod
    def install() -> None:
        """Placeholder install method.

        The real `tblib.pickling_support.install()` registers custom pickling
        handlers for traceback objects. In the test environment we don't need
        this functionality, so the method is intentionally a no‑op.
        """
        pass

# Expose the attribute expected by the codebase.
pickling_support = _PicklingSupport()

# Provide a minimal Traceback placeholder in case it's accessed.
class Traceback:
    def __init__(self, *args, **kwargs):
        pass

__all__ = ["pickling_support", "Traceback"]
