# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script

import contextlib
import dataclasses
import sys
import traceback
from typing import Callable, Generator


@dataclasses.dataclass
class BlameResult:
    found: bool = False
    trace_stack: str = ""


@contextlib.contextmanager
def blame(func: Callable) -> Generator[BlameResult, None, None]:
    """
    Trace the function calls to find the first function that satisfies the
    condition. The trace stack will be stored in the result.

    Usage:

    ```python
    with blame(lambda: some_condition()) as result:
        # do something
    
    if result.found:
        print(result.trace_stack)
    """
    result = BlameResult()

    def _trace_calls(frame, event, arg=None):
        nonlocal result
        if event in ['call', 'return']:
            # for every function call or return
            try:
                # Temporarily disable the trace function
                sys.settrace(None)
                # check condition here
                if not result.found and func():
                    result.found = True
                    result.trace_stack = "".join(traceback.format_stack())
                # Re-enable the trace function
                sys.settrace(_trace_calls)
            except NameError:
                # modules are deleted during shutdown
                pass
        return _trace_calls

    sys.settrace(_trace_calls)

    yield result

    sys.settrace(None)


module_name = "torch._inductor.async_compile"

with blame(lambda: module_name in sys.modules) as result:
    import vllm  # noqa

assert not result.found, (f"Module {module_name} is already imported, the"
                          f" first import location is:\n{result.trace_stack}")

print(f"Module {module_name} is not imported yet")
