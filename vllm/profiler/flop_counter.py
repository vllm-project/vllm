# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

from torch.utils.flop_counter import FlopCounterMode

__all__ = ["FlopContextManager", "format_flops"]


@dataclass
class FlopCount:
    """Container for FLOP counts for different operation types."""
    total_flops: int = 0
    flop_counts: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        """Return total FLOP count."""
        return self.total_flops

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {"total_flops": self.total_flops, **self.flop_counts}

    def __add__(self, other: 'FlopCount') -> 'FlopCount':
        """Add two FlopCount objects."""
        result_counts = self.flop_counts.copy()
        for op, count in other.flop_counts.items():
            result_counts[op] = result_counts.get(op, 0) + count

        return FlopCount(total_flops=self.total_flops + other.total_flops,
                         flop_counts=result_counts)

    def __iadd__(self, other: 'FlopCount') -> 'FlopCount':
        """In-place addition of FlopCount objects."""
        self.total_flops += other.total_flops
        for op, count in other.flop_counts.items():
            self.flop_counts[op] = self.flop_counts.get(op, 0) + count
        return self


@dataclass
class DetailedFlopCount:
    """Detailed FLOP counts organized by operation type and layer."""
    operation_counts: dict[str, int] = field(default_factory=dict)
    total_flops: int = 0

    def add_operation(self, op_name: str, flops: int):
        """Add FLOP count for a specific operation.
        
        Args:
            op_name: Name of the operation (e.g., 'aten::mm').
            flops: Number of floating point operations.
        """
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops


class FlopCounter:
    """A wrapper around PyTorch's FlopCounterMode for FLOP counting.
    
    This class provides a compatible interface to PyTorch's official
    FlopCounterMode while maintaining backward compatibility with the
    existing vLLM API.
    """

    def __init__(self, display: bool = False):
        """Initialize FlopCounter.
        
        Args:
            display: Whether to print FLOP statistics automatically.
        """
        self._flop_mode = FlopCounterMode(display=display)
        self._detailed_counts = DetailedFlopCount()

    def get_total_flops(self) -> int:
        """Get total FLOP count."""
        return self._flop_mode.get_total_flops()

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get FLOP breakdown by operation type."""
        return self._flop_mode.get_flop_counts()

    def get_detailed_counts(self) -> DetailedFlopCount:
        """Get detailed FLOP counts."""
        # Update with current data from FlopCounterMode
        self._detailed_counts.total_flops = self.get_total_flops()
        self._detailed_counts.operation_counts = self.get_flop_breakdown()
        return self._detailed_counts

    def reset(self):
        """Reset all counters."""
        # Create a new FlopCounterMode instance to reset
        self._flop_mode = FlopCounterMode(display=False)
        self._detailed_counts = DetailedFlopCount()

    def get_table(self) -> str:
        """Get formatted table of FLOP statistics."""
        return self._flop_mode.get_table()

    def __enter__(self):
        self._flop_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._flop_mode.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def FlopContextManager():
    """Context manager for FLOP counting.
    
    Usage:
        with FlopContextManager() as flop_counter:
            # Your model operations here
            outputs = model(inputs)
        
        print(f"Total FLOPs: {flop_counter.get_total_flops()}")
        print(f"Breakdown: {flop_counter.get_flop_breakdown()}")
    """
    counter = FlopCounter()
    with counter:
        yield counter


def format_flops(flops: int) -> str:
    """Format FLOP count in human-readable units.
    
    Args:
        flops: Number of floating point operations.
    """
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
