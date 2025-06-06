# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

from torch.utils.flop_counter import FlopCounterMode

__all__ = ["FlopContextManager", "format_flops"]


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

    def add_operation(self, op_name: str, flops: int):
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops


class FlopCounter:

    def __init__(self, display: bool = False):
        self._display = display
        self._flop_mode: FlopCounterMode | None = None
        self._detailed_counts = DetailedFlopCount()

    def get_total_flops(self) -> int:
        return self._flop_mode.get_total_flops()

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get categorized FLOP breakdown."""
        raw_flops = self._flop_mode.get_flop_counts()

        # Extract operations from the 'Global' module which contains
        # aggregated counts
        global_flops = raw_flops.get('Global', {})

        mm_flops = 0
        attention_flops = 0
        activation_flops = 0
        normalization_flops = 0

        for op, count in global_flops.items():
            op_name = str(op)
            if 'mm' in op_name or 'bmm' in op_name or 'addmm' in op_name:
                mm_flops += count
            elif 'attention' in op_name:
                attention_flops += count
            elif any(activation in op_name
                     for activation in ['relu', 'gelu', 'silu', 'swish']):
                activation_flops += count
            elif any(norm in op_name
                     for norm in ['layer_norm', 'group_norm', 'rms_norm']):
                normalization_flops += count

        return {
            'mm_flops': mm_flops,
            'attention_flops': attention_flops,
            'activation_flops': activation_flops,
            'normalization_flops': normalization_flops
        }

    def get_detailed_counts(self) -> DetailedFlopCount:
        raw_flops = self._flop_mode.get_flop_counts()
        global_flops = raw_flops.get('Global', {})

        self._detailed_counts.total_flops = self.get_total_flops()
        self._detailed_counts.operation_counts = global_flops

        # Get categorized breakdown
        breakdown = self.get_flop_breakdown()
        self._detailed_counts.mm_flops = breakdown['mm_flops']
        self._detailed_counts.attention_flops = breakdown['attention_flops']
        self._detailed_counts.activation_flops = (
            breakdown['activation_flops'])
        self._detailed_counts.normalization_flops = (
            breakdown['normalization_flops'])

        return self._detailed_counts

    def reset(self):
        self._detailed_counts = DetailedFlopCount()

    def get_table(self) -> str:
        return self._flop_mode.get_table()

    def __enter__(self):
        self._flop_mode = FlopCounterMode(display=self._display)
        self._flop_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._flop_mode.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def FlopContextManager():
    counter = FlopCounter()
    with counter:
        yield counter


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
