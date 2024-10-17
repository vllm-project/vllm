import dataclasses
from typing import Callable, Dict, List, Type, Union

from torch._C._profiler import _EventType, _ProfilerEvent, _TensorMetadata

#
# String / Print Manipulation
#


def trim_string_front(string, width):
    if len(string) > width:
        offset = len(string) - width + 3
        string = string[offset:]
        if len(string) > 3:
            string = "..." + string[3:]
    return string


def trim_string_back(string, width):
    if len(string) > width:
        offset = len(string) - width + 3
        string = string[:-offset]
        if len(string) > 3:
            string = string + "..."
    return string


class TablePrinter:

    def __init__(self, row_cls: Type[dataclasses.dataclass],
                 column_widths: Dict[str, int]):
        self.row_cls = row_cls
        self.fieldnames = [x.name for x in dataclasses.fields(row_cls)]
        self.column_widths = column_widths
        assert set(self.column_widths.keys()) == set(self.fieldnames)

    def print_table(self, rows: List[dataclasses.dataclass]):
        self._print_header()
        self._print_line()
        for row in rows:
            self._print_row(row)

    def _print_header(self):
        for i, f in enumerate(self.fieldnames):
            last = (i == len(self.fieldnames) - 1)
            col_width = self.column_widths[f]
            print(trim_string_back(f, col_width).ljust(col_width),
                  end=" | " if not last else "\n")

    def _print_row(self, row):
        assert isinstance(row, self.row_cls)

        for i, f in enumerate(self.fieldnames):
            last = (i == len(self.fieldnames) - 1)
            col_width = self.column_widths[f]
            val = getattr(row, f)

            val_str = ""
            if isinstance(val, str):
                val_str = trim_string_back(val, col_width).ljust(col_width)
            elif type(val) in [float, int]:
                val_str = f"{float(val):>.2f}".rjust(col_width)
            else:
                val_str = f"{val}".rjust(col_width)
            print(val_str, end=" | " if not last else "\n")

    def _print_line(self):
        total_col_width = 0
        for column_width in self.column_widths.values():
            total_col_width += column_width
        print("=" * (total_col_width + 3 * (len(self.column_widths) - 1)))


def indent_string(string: str,
                  indent: int,
                  indent_style: Union[Callable[[int], str], str] = " ") -> str:
    if indent:
        if isinstance(indent_style, str):
            return indent_style * indent + string
        else:
            return indent_style(indent) + string
    else:
        return string


#
# _ProfilerEvent utils
#


def event_has_module(event: _ProfilerEvent) -> bool:
    event_type, typed_event = event.typed
    if event_type == _EventType.PyCall:
        return typed_event.module is not None
    return False


def event_is_torch_op(event: _ProfilerEvent) -> bool:
    return event.tag == _EventType.TorchOp


def event_arg_repr(arg) -> str:
    if arg is None or type(arg) in [float, int, bool, str]:
        return f"{arg}"
    elif isinstance(arg, list):
        return f"[{', '.join([event_arg_repr(x) for x in arg])}]"
    elif isinstance(arg, tuple):
        return f"({', '.join([event_arg_repr(x) for x in arg])})"
    else:
        assert isinstance(arg,
                          _TensorMetadata), f"Unsupported type: {type(arg)}"
        sizes_str = ', '.join([str(x) for x in arg.sizes])
        return f"{str(arg.dtype).replace('torch.', '')}[{sizes_str}]"


def event_torch_op_repr(event: _ProfilerEvent) -> str:
    assert event.tag == _EventType.TorchOp
    args_str = ', '.join([event_arg_repr(x) for x in event.typed[1].inputs])
    return f"{event.name}({args_str})".replace("aten::", "")


def event_module_repr(event: _ProfilerEvent) -> str:
    assert event_has_module(event)
    module = event.typed[1].module
    if module.parameters and len(module.parameters) > 0:
        args_str = ', '.join(
            [f'{x[0]}={event_arg_repr(x[1])}' for x in module.parameters])
        return f"{module.cls_name}({args_str})"
    else:
        return module.cls_name


def event_torch_op_stack_trace(curr_event: _ProfilerEvent,
                               until: Callable[[_ProfilerEvent], bool]) -> str:
    trace = ""
    curr_event = curr_event.parent
    while curr_event and not until(curr_event):
        if event_is_torch_op(curr_event):
            if len(trace) > 0:
                trace += " <- "
            trace += event_torch_op_repr(curr_event)
        curr_event = curr_event.parent

    return trace
