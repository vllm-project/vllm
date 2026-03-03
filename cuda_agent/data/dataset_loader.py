"""
data/dataset_loader.py

Loads and preprocesses the CUDA-Agent-Ops-6K dataset from HuggingFace.

Dataset: BytedTsinghua-SIA/CUDA-Agent-Ops-6K
  - 6,000 training samples of PyTorch operator optimization tasks
  - Fields: ops (list of op names), data_source (str), code (str)
  - Each code sample is a self-contained PyTorch Model class

Usage:
    from cuda_agent.data.dataset_loader import load_cuda_agent_dataset
    dataset = load_cuda_agent_dataset()
    for sample in dataset:
        print(sample["ops"], sample["code"][:200])
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CUDAAgentSample:
    """A single training sample from CUDA-Agent-Ops-6K."""
    ops: list[str]           # e.g. ["nn.Conv2d", "torch.relu"]
    data_source: str         # e.g. "torch#2"
    code: str                # Full Model class Python source
    sample_id: str = field(default="")

    def __post_init__(self):
        if not self.sample_id:
            h = hashlib.md5(self.code.encode()).hexdigest()[:8]
            self.sample_id = f"{self.data_source}_{h}"

    @property
    def ops_str(self) -> str:
        return ", ".join(self.ops)

    def to_task_description(self) -> str:
        """Human-readable problem statement fed to the CUDA agent."""
        return (
            f"# CUDA Kernel Optimization Task\n\n"
            f"**Operations used:** {self.ops_str}\n\n"
            f"Your goal is to accelerate the following PyTorch model by "
            f"replacing its operations with custom CUDA kernels. "
            f"The optimized implementation must achieve at least 5% speedup "
            f"over `torch.compile` while maintaining correctness "
            f"(atol=1e-2, rtol=1e-2).\n\n"
            f"```python\n{self.code}\n```\n\n"
            f"Read SKILL.md for workflow constraints and optimization guidelines."
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DATASET_NAME = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def load_cuda_agent_dataset(
    split: str = "train",
    cache_dir: str | Path | None = None,
    max_samples: int | None = None,
    filter_ops: list[str] | None = None,
    min_code_len: int = 275,
    max_code_len: int = 14_100,
) -> list[CUDAAgentSample]:
    """
    Load and preprocess the CUDA-Agent-Ops-6K dataset.

    Args:
        split:        Dataset split (only "train" is available).
        cache_dir:    Local cache directory for HuggingFace datasets.
        max_samples:  Cap the number of samples returned (None = all 6,000).
        filter_ops:   If given, only return samples whose ops overlap with
                      this list (useful for curriculum ordering).
        min_code_len: Minimum code length in characters.
        max_code_len: Maximum code length in characters.

    Returns:
        List of CUDAAgentSample objects.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets"
        ) from exc

    logger.info("Loading dataset %s (split=%s) …", DATASET_NAME, split)
    raw = load_dataset(
        DATASET_NAME,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    samples: list[CUDAAgentSample] = []
    for row in raw:
        ops = _parse_ops(row["ops"])
        code = row["code"]
        data_source = row.get("data_source", "")

        # Basic length filter
        if not (min_code_len <= len(code) <= max_code_len):
            continue

        # Optional op-set filter (intersection check)
        if filter_ops is not None:
            if not any(op in ops for op in filter_ops):
                continue

        samples.append(CUDAAgentSample(
            ops=ops,
            data_source=data_source,
            code=code,
        ))

        if max_samples is not None and len(samples) >= max_samples:
            break

    logger.info("Loaded %d samples from %s.", len(samples), DATASET_NAME)
    return samples


def _parse_ops(raw_ops) -> list[str]:
    """Normalise the ops field: handles string repr of list or actual list."""
    if isinstance(raw_ops, list):
        return [str(o) for o in raw_ops]
    if isinstance(raw_ops, str):
        try:
            parsed = ast.literal_eval(raw_ops)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except (ValueError, SyntaxError):
            pass
        # Fallback: comma-separated
        return [o.strip() for o in raw_ops.split(",") if o.strip()]
    return []


# ---------------------------------------------------------------------------
# Curriculum ordering (matches paper: sort by num_ops then code complexity)
# ---------------------------------------------------------------------------

def sort_by_difficulty(samples: list[CUDAAgentSample]) -> list[CUDAAgentSample]:
    """
    Order samples from simplest to most complex (curriculum learning).
    Primary key: number of operations; secondary key: code length.
    """
    return sorted(samples, key=lambda s: (len(s.ops), len(s.code)))


def split_by_num_ops(
    samples: list[CUDAAgentSample],
) -> dict[int, list[CUDAAgentSample]]:
    """Group samples by number of operator compositions."""
    groups: dict[int, list[CUDAAgentSample]] = {}
    for s in samples:
        n = len(s.ops)
        groups.setdefault(n, []).append(s)
    return groups


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def dataset_statistics(samples: list[CUDAAgentSample]) -> dict:
    """Compute summary statistics matching the paper's dataset analysis."""
    from collections import Counter
    op_counts = Counter(len(s.ops) for s in samples)
    source_counts = Counter(s.data_source for s in samples)
    code_lengths = [len(s.code) for s in samples]

    return {
        "total_samples": len(samples),
        "num_ops_distribution": dict(op_counts),
        "source_distribution": dict(source_counts),
        "code_length_stats": {
            "min": min(code_lengths),
            "max": max(code_lengths),
            "mean": sum(code_lengths) / len(code_lengths),
        },
        # Paper: 83.77% of samples are 2-operator compositions
        "two_op_fraction": op_counts.get(2, 0) / len(samples) if samples else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    samples = load_cuda_agent_dataset(max_samples=max_n)
    stats = dataset_statistics(samples)
    print(json.dumps(stats, indent=2))
    print("\nSample 0:")
    print(samples[0].to_task_description()[:800] if samples else "No samples.")
