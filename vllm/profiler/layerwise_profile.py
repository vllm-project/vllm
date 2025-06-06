# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, TypeAlias, Union

import pandas as pd
from torch._C._autograd import DeviceType, _KinetoEvent, _ProfilerResult
from torch._C._profiler import _EventType, _ExperimentalConfig, _ProfilerEvent
from torch.autograd.profiler import FunctionEvent
from torch.profiler import ProfilerActivity, profile

from vllm.profiler.flop_counter import (DetailedFlopCount, FlopCounter,
                                        format_flops)
from vllm.profiler.utils import (TablePrinter, event_has_module,
                                 event_is_torch_op, event_module_repr,
                                 event_torch_op_stack_trace, indent_string)


@dataclass
class _ModuleTreeNode:
    event: _ProfilerEvent
    parent: Optional['_ModuleTreeNode'] = None
    children: list['_ModuleTreeNode'] = field(default_factory=list)
    trace: str = ""

    @property
    def is_leaf(self):
        return (self.event.children is None or len(self.event.children) == 0)

    @property
    def is_torch_op(self):
        return event_is_torch_op(self.event)

    @property
    def is_cuda(self):
        return (self.event.tag == _EventType.Kineto
                and self.event.typed[1].device_type == DeviceType.CUDA)


@dataclass
class SummaryStatsEntry:
    name: str
    cuda_time_us: float
    pct_cuda_time: float
    invocations: int
    flops: int = 0
    gflops_per_sec: float = 0.0


@dataclass
class ModelStatsEntry:
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str
    flops: int = 0
    gflops_per_sec: float = 0.0


StatsEntry: TypeAlias = Union[ModelStatsEntry, SummaryStatsEntry]


@dataclass
class _StatsTreeNode:
    entry: StatsEntry
    children: list[StatsEntry]
    parent: Optional[StatsEntry]


@dataclass
class LayerwiseProfileResults(profile):
    _kineto_results: _ProfilerResult
    _kineto_event_correlation_map: dict[int,
                                        list[_KinetoEvent]] = field(init=False)
    _event_correlation_map: dict[int, list[FunctionEvent]] = field(init=False)
    _module_tree: list[_ModuleTreeNode] = field(init=False)
    _model_stats_tree: list[_StatsTreeNode] = field(init=False)
    _summary_stats_tree: list[_StatsTreeNode] = field(init=False)

    # profile metadata
    num_running_seqs: Optional[int] = None
    flop_counts: Optional[DetailedFlopCount] = None

    def __post_init__(self):
        self._build_correlation_map()
        self._build_module_tree()
        self._build_stats_trees()

    def print_model_table(self, column_widths: dict[str, int] = None):
        _column_widths = dict(name=60,
                              cpu_time_us=12,
                              cuda_time_us=12,
                              pct_cuda_time=12,
                              flops=15,
                              gflops_per_sec=15,
                              trace=60)
        if column_widths:
            _column_widths.update(**column_widths)
        filtered_model_table = [
            (depth, row)
            for depth, row in self._flatten_stats_tree(self._model_stats_tree)
            if row.cuda_time_us > 0 or row.cpu_time_us > 0
        ]
        TablePrinter(ModelStatsEntry, _column_widths).print_table(
            self._indent_row_names_based_on_depth(
                filtered_model_table,
                indent_style=lambda indent: "|" + "-" * indent + " "))

    def print_summary_table(self, column_widths: dict[str, int] = None):
        _column_widths = dict(name=80,
                              cuda_time_us=12,
                              pct_cuda_time=12,
                              invocations=15,
                              flops=15,
                              gflops_per_sec=15)
        if column_widths:
            _column_widths.update(**column_widths)
        filtered_summary_table = [(depth, row)
                                  for depth, row in self._flatten_stats_tree(
                                      self._summary_stats_tree)
                                  if row.cuda_time_us > 0]
        TablePrinter(SummaryStatsEntry, _column_widths).print_table(
            self._indent_row_names_based_on_depth(
                filtered_summary_table,
                indent_style=lambda indent: "|" + "-" * indent + " "))

    def export_model_stats_table_csv(self, filename: str):
        df = pd.DataFrame([
            asdict(row)
            for _, row in self._flatten_stats_tree(self._model_stats_tree)
        ])
        df.to_csv(filename)

    def export_summary_stats_table_csv(self, filename: str):
        df = pd.DataFrame([
            asdict(row)
            for _, row in self._flatten_stats_tree(self._summary_stats_tree)
        ])
        df.to_csv(filename)

    def print_flop_summary(self):
        """Print a summary of FLOP counts."""
        if not self.flop_counts:
            print("No FLOP data available")
            return

        print("\n=== FLOP Summary ===")
        print(f"Total FLOPs: {format_flops(self.flop_counts.total_flops)}")

        if self.flop_counts.operation_counts:
            print("\nTop Operations by FLOP Count:")
            sorted_ops = sorted(self.flop_counts.operation_counts.items(),
                                key=lambda x: x[1],
                                reverse=True)
            for op_name, flops in sorted_ops[:10]:  # Top 10
                print(f"  {op_name}: {format_flops(flops)}")

        if self.flop_counts.layer_counts:
            print("\nTop Layers by FLOP Count:")
            layer_totals = [
                (layer, flop_count.total())
                for layer, flop_count in self.flop_counts.layer_counts.items()
            ]
            sorted_layers = sorted(layer_totals,
                                   key=lambda x: x[1],
                                   reverse=True)
            for layer_name, total_flops in sorted_layers[:10]:  # Top 10
                print(f"  {layer_name}: {format_flops(total_flops)}")

    def convert_stats_to_dict(self) -> dict[str, Any]:
        result = {
            "metadata": {
                "num_running_seqs": self.num_running_seqs
            },
            "summary_stats":
            self._convert_stats_tree_to_dict(self._summary_stats_tree),
            "model_stats":
            self._convert_stats_tree_to_dict(self._model_stats_tree)
        }

        if self.flop_counts:
            result["flop_summary"] = {
                "total_flops": self.flop_counts.total_flops,
                "operation_counts": self.flop_counts.operation_counts,
                "layer_counts": {
                    layer: flop_count.to_dict()
                    for layer, flop_count in
                    self.flop_counts.layer_counts.items()
                }
            }

        return result

    @staticmethod
    def _indent_row_names_based_on_depth(depths_rows: list[tuple[int,
                                                                 StatsEntry]],
                                         indent_style: Union[Callable[[int],
                                                                      str],
                                                             str] = " "):
        indented_rows = []
        for depth, row in depths_rows:
            if row.cuda_time_us == 0:
                continue
            indented_row = copy.deepcopy(row)
            indented_row.name = indent_string(indented_row.name, depth,
                                              indent_style)
            indented_rows.append(indented_row)
        return indented_rows

    def _build_correlation_map(self):
        self._kineto_event_correlation_map = defaultdict(list)
        for event in self._kineto_results.events():
            self._kineto_event_correlation_map[event.correlation_id()].append(
                event)

    def _build_module_tree(self):
        self._module_tree = []
        event_tree = self._kineto_results.experimental_event_tree()

        def _df_traversal(event: _ProfilerEvent,
                          curr_node: Optional[_ModuleTreeNode] = None):

            # For the tensor parallel case for now only look at task 1
            if event.start_tid != 1:
                return

            if event_has_module(event):
                node = _ModuleTreeNode(event=event, parent=curr_node)
                if curr_node:
                    curr_node.children.append(node)
                else:
                    self._module_tree.append(node)
                curr_node = node

            is_leaf = (event.children is None or len(event.children) == 0)
            if is_leaf and curr_node:
                node = _ModuleTreeNode(
                    event=event,
                    parent=curr_node,
                    trace=event_torch_op_stack_trace(
                        event, until=lambda x: event_has_module(x)))
                curr_node.children.append(node)
                curr_node = node

            for child in event.children:
                _df_traversal(child, curr_node)

        for root in event_tree:
            _df_traversal(root)

    def _get_kineto_gpu_event(self, node: _ModuleTreeNode):
        if node.event.tag != _EventType.Kineto:
            return None
        correlated_kineto_events = self._kineto_event_correlation_map.get(
            node.event.correlation_id, [])
        iterator = (x for x in correlated_kineto_events
                    if x.device_type() == DeviceType.CUDA
                    and x.name() == node.event.name)
        return next(iterator, None)

    def _cumulative_cuda_time(self, node: _ModuleTreeNode):
        'Return cuda time in microseconds'

        def _cumulative_cuda_time_recursive(node: _ModuleTreeNode):
            if node.is_leaf and (gpu_kineto_event :=
                                 self._get_kineto_gpu_event(node)):
                return gpu_kineto_event.duration_ns() / 1000.0
            else:
                cumulative_cuda_time = 0
                for child in node.children:
                    cumulative_cuda_time += _cumulative_cuda_time_recursive(
                        child)
                return cumulative_cuda_time

        return _cumulative_cuda_time_recursive(node)

    def _total_cuda_time(self):
        return sum(
            [self._cumulative_cuda_time(root) for root in self._module_tree])

    def _get_flop_count_for_layer(self, layer_name: str) -> int:
        """Get FLOP count for a specific layer from flop_counts.
        
        Args:
            layer_name: Name of the layer to get FLOP counts for.
        """
        if not self.flop_counts or not self.flop_counts.layer_counts:
            return 0

        # First try exact match
        if layer_name in self.flop_counts.layer_counts:
            return self.flop_counts.layer_counts[layer_name].total()

        # Then try hierarchical prefix matching (for nested modules)
        for layer, flop_count in self.flop_counts.layer_counts.items():
            if layer_name.startswith(layer +
                                     ".") or layer.startswith(layer_name +
                                                              "."):
                return flop_count.total()

        return 0

    def _get_flop_count_for_operation(self, op_name: str) -> int:
        """Get FLOP count for a specific operation from flop_counts.
        
        Args:
            op_name: Name of the operation to get FLOP counts for.
        """
        if not self.flop_counts or not self.flop_counts.operation_counts:
            return 0

        # First try exact match
        if op_name in self.flop_counts.operation_counts:
            return self.flop_counts.operation_counts[op_name]

        # Then try suffix matching for operation overloads
        for op, flops in self.flop_counts.operation_counts.items():
            if op_name.endswith(op.split('.')[-1]) or op.endswith(
                    op_name.split('.')[-1]):
                return flops

        return 0

    def _calculate_gflops_per_sec(self, flops: int, time_us: float) -> float:
        """Calculate GFLOPS/sec given FLOP count and time in microseconds.
        
        Args:
            flops: Number of floating point operations.
            time_us: Time in microseconds.
        """
        if time_us == 0:
            return 0.0
        time_sec = time_us / 1_000_000
        return flops / (time_sec * 1e9)

    def _build_stats_trees(self):
        summary_dict: dict[str, _StatsTreeNode] = {}
        total_cuda_time = self._total_cuda_time()

        def pct_cuda_time(cuda_time_us):
            return (cuda_time_us / total_cuda_time) * 100

        def build_summary_stats_tree_df(
            node: _ModuleTreeNode,
            parent: Optional[_StatsTreeNode] = None,
            summary_trace: tuple[str] = ()):

            if event_has_module(node.event):
                name = event_module_repr(node.event)
                cuda_time_us = self._cumulative_cuda_time(node)
                flops = self._get_flop_count_for_layer(name)
            elif (gpu_kineto_event := self._get_kineto_gpu_event(node)):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
                flops = self._get_flop_count_for_operation(name)
            else:
                return None

            gflops_per_sec = self._calculate_gflops_per_sec(
                flops, cuda_time_us)
            summary_trace = summary_trace + (name, )
            if summary_trace in summary_dict:
                entry = summary_dict[summary_trace].entry
                entry.cuda_time_us += cuda_time_us
                entry.flops += flops
                entry.invocations += 1
                entry.pct_cuda_time = pct_cuda_time(entry.cuda_time_us)
                entry.gflops_per_sec = self._calculate_gflops_per_sec(
                    entry.flops, entry.cuda_time_us)
            else:
                new_node = _StatsTreeNode(entry=SummaryStatsEntry(
                    name=name,
                    cuda_time_us=cuda_time_us,
                    pct_cuda_time=pct_cuda_time(cuda_time_us),
                    invocations=1,
                    flops=flops,
                    gflops_per_sec=gflops_per_sec),
                                          children=[],
                                          parent=parent)
                if parent:
                    parent.children.append(new_node)
                summary_dict[summary_trace] = new_node

            for child in node.children:
                build_summary_stats_tree_df(child, summary_dict[summary_trace],
                                            summary_trace)

            return summary_dict[summary_trace]

        self._summary_stats_tree = []
        for root in self._module_tree:
            self._summary_stats_tree.append(build_summary_stats_tree_df(root))

        def build_model_stats_tree_df(node: _ModuleTreeNode,
                                      parent: Optional[_StatsTreeNode] = None):
            if event_has_module(node.event, ):
                name = event_module_repr(node.event)
                cuda_time_us = self._cumulative_cuda_time(node)
                cpu_time_us = node.event.duration_time_ns / 1000
                trace = ""
                flops = self._get_flop_count_for_layer(name)
            elif (gpu_kineto_event := self._get_kineto_gpu_event(node)):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
                cpu_time_us = 0
                trace = node.trace
                flops = self._get_flop_count_for_operation(name)
            else:
                return None

            gflops_per_sec = self._calculate_gflops_per_sec(
                flops, cuda_time_us)
            new_node = _StatsTreeNode(entry=ModelStatsEntry(
                name=name,
                cpu_time_us=cpu_time_us,
                cuda_time_us=cuda_time_us,
                pct_cuda_time=pct_cuda_time(cuda_time_us),
                trace=trace,
                flops=flops,
                gflops_per_sec=gflops_per_sec),
                                      parent=parent,
                                      children=[])
            if parent:
                parent.children.append(new_node)

            for child in node.children:
                build_model_stats_tree_df(child, new_node)

            return new_node

        self._model_stats_tree = []
        for root in self._module_tree:
            self._model_stats_tree.append(build_model_stats_tree_df(root))

    def _flatten_stats_tree(
            self, tree: list[_StatsTreeNode]) -> list[tuple[int, StatsEntry]]:
        entries: list[tuple[int, StatsEntry]] = []

        def df_traversal(node: _StatsTreeNode, depth=0):
            entries.append((depth, node.entry))
            for child in node.children:
                df_traversal(child, depth=depth + 1)

        for root in tree:
            df_traversal(root)

        return entries

    def _convert_stats_tree_to_dict(self,
                                    tree: list[_StatsTreeNode]) -> list[dict]:
        root_dicts: list[dict] = []

        def df_traversal(node: _StatsTreeNode, curr_json_list: list[dict]):
            curr_json_list.append({
                "entry": asdict(node.entry),
                "children": []
            })
            for child in node.children:
                df_traversal(child, curr_json_list[-1]["children"])

        for root in tree:
            df_traversal(root, root_dicts)

        return root_dicts


class layerwise_profile(profile):

    def __init__(self,
                 num_running_seqs: Optional[int] = None,
                 enable_flop_counting: bool = False):
        """Layerwise profile constructor.

        Args:
            num_running_seqs: When given, num_running_seqs will be passed to 
                LayerProfileResults for metadata update.
            enable_flop_counting: Whether to enable FLOP counting during 
                profiling.
        """
        super().__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            experimental_config=_ExperimentalConfig(verbose=True))

        self.num_running_seqs = num_running_seqs
        self.enable_flop_counting = enable_flop_counting
        self.flop_counter: Optional[FlopCounter] = None

    def __enter__(self):
        if self.enable_flop_counting:
            self.flop_counter = FlopCounter()
            self.flop_counter.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        flop_counts = None
        if self.flop_counter:
            self.flop_counter.__exit__(exc_type, exc_val, exc_tb)
            flop_counts = self.flop_counter.get_detailed_counts()

        super().__exit__(exc_type, exc_val, exc_tb)
        self.results = LayerwiseProfileResults(
            self.profiler.kineto_results,
            num_running_seqs=self.num_running_seqs,
            flop_counts=flop_counts)
