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


@dataclass
class ModelStatsEntry:
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str


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

    def __post_init__(self):
        self._build_correlation_map()
        self._build_module_tree()
        self._build_stats_trees()

    def print_model_table(self, column_widths: dict[str, int] = None):
        _column_widths = dict(name=60,
                              cpu_time_us=12,
                              cuda_time_us=12,
                              pct_cuda_time=12,
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
                              invocations=15)
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

    def convert_stats_to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "num_running_seqs": self.num_running_seqs
            },
            "summary_stats":
            self._convert_stats_tree_to_dict(self._summary_stats_tree),
            "model_stats":
            self._convert_stats_tree_to_dict(self._model_stats_tree)
        }

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
            elif (gpu_kineto_event := self._get_kineto_gpu_event(node)):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
            else:
                return None

            summary_trace = summary_trace + (name, )
            if summary_trace in summary_dict:
                entry = summary_dict[summary_trace].entry
                entry.cuda_time_us += cuda_time_us
                entry.invocations += 1
                entry.pct_cuda_time = pct_cuda_time(entry.cuda_time_us)
            else:
                new_node = _StatsTreeNode(entry=SummaryStatsEntry(
                    name=name,
                    cuda_time_us=cuda_time_us,
                    pct_cuda_time=pct_cuda_time(cuda_time_us),
                    invocations=1),
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
            elif (gpu_kineto_event := self._get_kineto_gpu_event(node)):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
                cpu_time_us = 0
                trace = node.trace
            else:
                return None

            new_node = _StatsTreeNode(entry=ModelStatsEntry(
                name=name,
                cpu_time_us=cpu_time_us,
                cuda_time_us=cuda_time_us,
                pct_cuda_time=pct_cuda_time(cuda_time_us),
                trace=trace),
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

    def __init__(self, num_running_seqs: Optional[int] = None):
        """
        layerwise profile constructor.

        Args:
            num_running_seqs (Optional[int], optional): When given,
            num_running_seqs will be passed to LayerProfileResults for metadata
            update. Defaults to None.
        """
        super().__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            experimental_config=_ExperimentalConfig(verbose=True))

        self.num_running_seqs = num_running_seqs

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.results = LayerwiseProfileResults(
            self.profiler.kineto_results,
            num_running_seqs=self.num_running_seqs)
