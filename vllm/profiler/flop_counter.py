# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "FlopCounter", "FlopCount", "DetailedFlopCount", "FlopContextManager",
    "format_flops"
]


@dataclass
class FlopCount:
    total_flops: int = 0
    flop_counts: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return self.total_flops

    def to_dict(self) -> dict[str, int]:
        return {"total_flops": self.total_flops, **self.flop_counts}

    def __add__(self, other: 'FlopCount') -> 'FlopCount':
        result_counts = self.flop_counts.copy()
        for op, count in other.flop_counts.items():
            result_counts[op] = result_counts.get(op, 0) + count

        return FlopCount(total_flops=self.total_flops + other.total_flops,
                         flop_counts=result_counts)

    def __iadd__(self, other: 'FlopCount') -> 'FlopCount':
        self.total_flops += other.total_flops
        for op, count in other.flop_counts.items():
            self.flop_counts[op] = self.flop_counts.get(op, 0) + count
        return self


@dataclass
class DetailedFlopCount:
    operation_counts: dict[str, int] = field(default_factory=dict)
    total_flops: int = 0
    mm_flops: int = 0
    attention_flops: int = 0
    activation_flops: int = 0
    normalization_flops: int = 0
    # Additional categorizations for offline analysis
    embedding_flops: int = 0
    convolution_flops: int = 0
    other_flops: int = 0

    def add_operation(self, op_name: str, flops: int):
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops

    def get_breakdown_dict(self) -> dict[str, int]:
        """Get a dictionary breakdown of FLOP categories."""
        return {
            'total_flops': self.total_flops,
            'mm_flops': self.mm_flops,
            'attention_flops': self.attention_flops,
            'activation_flops': self.activation_flops,
            'normalization_flops': self.normalization_flops,
            'embedding_flops': self.embedding_flops,
            'convolution_flops': self.convolution_flops,
            'other_flops': self.other_flops
        }

    def get_percentage_breakdown(self) -> dict[str, float]:
        """Get percentage breakdown of FLOP categories."""
        if self.total_flops == 0:
            return {k: 0.0 for k in self.get_breakdown_dict()}

        breakdown = self.get_breakdown_dict()
        return {
            k: (v / self.total_flops * 100.0) if k != 'total_flops' else 100.0
            for k, v in breakdown.items()
        }


class FlopCounter:

    def __init__(self, display: bool = False):
        self._display = display
        self._flop_mode: FlopCounterMode | None = None
        self._detailed_counts = DetailedFlopCount()

    def get_total_flops(self) -> int:
        if self._flop_mode is None:
            return 0
        return self._flop_mode.get_total_flops()

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get categorized FLOP breakdown with enhanced categorization."""
        if self._flop_mode is None:
            return {
                'mm_flops': 0,
                'attention_flops': 0,
                'activation_flops': 0,
                'normalization_flops': 0,
                'embedding_flops': 0,
                'convolution_flops': 0,
                'other_flops': 0
            }
        raw_flops = self._flop_mode.get_flop_counts()

        # Extract operations from the 'Global' module which contains
        # aggregated counts
        global_flops = raw_flops.get('Global', {})

        mm_flops = 0
        attention_flops = 0
        activation_flops = 0
        normalization_flops = 0
        embedding_flops = 0
        convolution_flops = 0
        other_flops = 0

        for op, count in global_flops.items():
            op_name = str(op).lower()
            if any(mm_op in op_name
                   for mm_op in ['mm', 'bmm', 'addmm', 'matmul']):
                mm_flops += count
            elif 'attention' in op_name or 'attn' in op_name:
                attention_flops += count
            elif any(activation in op_name for activation in
                     ['relu', 'gelu', 'silu', 'swish', 'tanh', 'sigmoid']):
                activation_flops += count
            elif any(norm in op_name for norm in
                     ['layer_norm', 'group_norm', 'rms_norm', 'batch_norm']):
                normalization_flops += count
            elif 'embedding' in op_name or 'embed' in op_name:
                embedding_flops += count
            elif any(conv_op in op_name
                     for conv_op in ['conv', 'convolution']):
                convolution_flops += count
            else:
                other_flops += count

        return {
            'mm_flops': mm_flops,
            'attention_flops': attention_flops,
            'activation_flops': activation_flops,
            'normalization_flops': normalization_flops,
            'embedding_flops': embedding_flops,
            'convolution_flops': convolution_flops,
            'other_flops': other_flops
        }

    def get_detailed_counts(self) -> DetailedFlopCount:
        if self._flop_mode is None:
            return self._detailed_counts

        raw_flops = self._flop_mode.get_flop_counts()
        global_flops = raw_flops.get('Global', {})

        self._detailed_counts.total_flops = self.get_total_flops()
        self._detailed_counts.operation_counts = global_flops

        # Get categorized breakdown
        breakdown = self.get_flop_breakdown()
        self._detailed_counts.mm_flops = breakdown['mm_flops']
        self._detailed_counts.attention_flops = breakdown['attention_flops']
        self._detailed_counts.activation_flops = breakdown['activation_flops']
        self._detailed_counts.normalization_flops = (
            breakdown['normalization_flops'])
        self._detailed_counts.embedding_flops = breakdown['embedding_flops']
        self._detailed_counts.convolution_flops = breakdown[
            'convolution_flops']
        self._detailed_counts.other_flops = breakdown['other_flops']

        return self._detailed_counts

    def get_efficiency_metrics(self,
                               elapsed_time_sec: float) -> dict[str, float]:
        """Calculate efficiency metrics for offline analysis."""
        total_flops = self.get_total_flops()
        if elapsed_time_sec <= 0 or total_flops == 0:
            return {
                'gflops_per_sec': 0.0,
                'tflops_per_sec': 0.0,
                'flops_per_microsec': 0.0
            }

        return {
            'gflops_per_sec': total_flops / (elapsed_time_sec * 1e9),
            'tflops_per_sec': total_flops / (elapsed_time_sec * 1e12),
            'flops_per_microsec': total_flops / (elapsed_time_sec * 1e6)
        }

    def print_analysis_summary(self, elapsed_time_sec: float = None):
        """Print a comprehensive analysis summary for offline use."""
        total_flops = self.get_total_flops()
        breakdown = self.get_flop_breakdown()

        print("\n=== FLOP Analysis Summary ===")
        print(f"Total FLOPs: {format_flops(total_flops)}")

        if elapsed_time_sec:
            efficiency = self.get_efficiency_metrics(elapsed_time_sec)
            print(f"Elapsed Time: {elapsed_time_sec:.3f} seconds")
            print(
                f"Performance: {efficiency['gflops_per_sec']:.2f} GFLOPS/sec")
            print(
                f"Performance: {efficiency['tflops_per_sec']:.4f} TFLOPS/sec")

        print("\n=== FLOP Breakdown ===")
        for category, flops in breakdown.items():
            if flops > 0:
                percentage = ((flops / total_flops *
                               100) if total_flops > 0 else 0)
                flop_str = format_flops(flops)
                print(f"{category:20s}: {flop_str:>12s} ({percentage:5.1f}%)")

    def reset(self):
        self._detailed_counts = DetailedFlopCount()

    def get_table(self) -> str:
        if self._flop_mode is None:
            return "No FLOP data available"
        return self._flop_mode.get_table()

    def __enter__(self):
        self._flop_mode = FlopCounterMode(display=self._display)
        self._flop_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._flop_mode.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def FlopContextManager(display: bool = False, auto_print: bool = False):
    """Context manager for FLOP counting in offline analysis.
    
    Args:
        display: Whether to display detailed PyTorch FLOP table
        auto_print: Whether to automatically print analysis summary on exit
    """
    counter = FlopCounter(display=display)
    start_time = None

    try:
        import time
        start_time = time.time()
        with counter:
            yield counter
    finally:
        if auto_print and start_time is not None:
            elapsed_time = time.time() - start_time
            counter.print_analysis_summary(elapsed_time)


def format_flops(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"
