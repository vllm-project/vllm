import contextlib
import dataclasses
import sys
import traceback
from typing import Any, Callable, Generator, List


@dataclasses.dataclass
class MonitoredValues:
    values: List[Any] = dataclasses.field(default_factory=list)
    trace_stacks: List[str] = dataclasses.field(default_factory=list)


@contextlib.contextmanager
def monitor(
        measure_func: Callable[[],
                               Any]) -> Generator[MonitoredValues, None, None]:
    """
    Trace the function calls to continuously monitor the change of
    a value.

    Usage:

    ```python

    def measure_func():
        ... # measure the current value
        return current_value

    with monitor(measure_func) as monitored_values:
        # do something
    
    monitored_values.values # all changes of the values
    monitored_values.trace_stacks # trace stacks of every change
    """
    monitored_values = MonitoredValues()

    def _trace_calls(frame, event, arg=None):
        nonlocal monitored_values
        if event in ['call', 'return']:
            # for every function call or return
            try:
                # Temporarily disable the trace function
                sys.settrace(None)
                # do a measurement
                current_value = measure_func()
                if len(monitored_values.values
                       ) == 0 or current_value != monitored_values.values[-1]:
                    monitored_values.values.append(current_value)
                    monitored_values.trace_stacks.append("".join(
                        traceback.format_stack()))
                # Re-enable the trace function
                sys.settrace(_trace_calls)
            except NameError:
                # modules are deleted during shutdown
                pass
        return _trace_calls

    try:
        sys.settrace(_trace_calls)
        yield monitored_values
    finally:
        sys.settrace(None)
