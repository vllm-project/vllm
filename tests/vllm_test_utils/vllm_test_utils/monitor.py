# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import dataclasses
import sys
import traceback
from collections.abc import Generator
from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")


@dataclasses.dataclass
class MonitoredValues(Generic[_T]):
    values: list[_T] = dataclasses.field(default_factory=list)
    trace_stacks: list[str] = dataclasses.field(default_factory=list)


@contextlib.contextmanager
def monitor(
    measure_func: Callable[[],
                           _T]) -> Generator[MonitoredValues[_T], None, None]:
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
    ```
    """
    monitored_values = MonitoredValues[_T]()

    def _trace_calls(frame, event, arg=None):
        nonlocal monitored_values
        if event in ['line']:
            # triggered by every line of Python code.
            # only Python functions will trigger it,
            # c/cpp functions will not trigger it.
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
