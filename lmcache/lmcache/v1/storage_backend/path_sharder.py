# SPDX-License-Identifier: Apache-2.0
"""Shared path-sharding logic for multi-path storage backends.

A :class:`PathSharder` takes a comma-separated list of directory paths
and a sharding strategy, then selects a single path for the current
worker.  Both :class:`LocalDiskBackend` and :class:`GdsBackend` delegate
path selection to this module so the policy lives in one place.
"""

# Standard
from typing import List
import os

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _resolve_device_id(dst_device: str) -> int:
    """Derive an integer device index from *dst_device*.

    Args:
        dst_device: Device string such as ``"cuda:2"``, ``"cuda"``,
            or ``"cpu"``.

    Returns:
        Integer device index.  Falls back to
        :func:`torch_dev.current_device` when the string carries no
        explicit index, or ``0`` when the accelerator is unavailable.
    """
    if ":" in dst_device:
        try:
            return int(dst_device.split(":", 1)[1])
        except ValueError:
            logger.warning(f"Invalid device index in '{dst_device}', falling back.")
    if torch_dev.is_available() and dst_device != "cpu":
        return torch_dev.current_device()
    return 0


class PathSharder:
    """Select one path from a comma-separated list using a sharding policy.

    Args:
        raw_csv: Comma-separated directory paths (e.g.
            ``"/mnt/nvme0/cache,/mnt/nvme1/cache"``).
        strategy: Sharding strategy name.  Currently only ``"by_gpu"``
            is supported, which selects ``paths[device_id % len(paths)]``.
        dst_device: Device string used to derive the worker index
            (e.g. ``"cuda:0"``, ``"cuda"``, ``"cpu"``).
        create_dirs: If ``True``, create **all** directories in the
            list at construction time (not just the selected one).

    Raises:
        ValueError: If *raw_csv* is empty or contains no valid paths.
        ValueError: If *strategy* is not a supported sharding mode.

    Example::

        sharder = PathSharder(
            "/mnt/nvme0/cache,/mnt/nvme1/cache",
            strategy="by_gpu",
            dst_device="cuda:1",
        )
        sharder.selected   # "/mnt/nvme1/cache"
        sharder.all_paths  # ["/mnt/nvme0/cache", "/mnt/nvme1/cache"]
    """

    _SUPPORTED_STRATEGIES = ("by_gpu",)

    def __init__(
        self,
        raw_csv: str,
        strategy: str,
        dst_device: str,
        create_dirs: bool = False,
    ) -> None:
        paths = [p.strip() for p in raw_csv.split(",") if p.strip()]
        if not paths:
            raise ValueError("At least one path must be provided")

        if strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported path sharding strategy '{strategy}'. "
                f"Supported: {', '.join(self._SUPPORTED_STRATEGIES)}"
            )

        device_id = _resolve_device_id(dst_device)

        self._all_paths: List[str] = paths
        self._strategy: str = strategy
        self._selected: str = paths[device_id % len(paths)]

        if create_dirs:
            for p in paths:
                os.makedirs(p, exist_ok=True)

    # -- public read-only properties -----------------------------------------

    @property
    def selected(self) -> str:
        """The single path chosen for this worker."""
        return self._selected

    @property
    def all_paths(self) -> List[str]:
        """All configured paths (unmodified order)."""
        return list(self._all_paths)

    @property
    def strategy(self) -> str:
        """Name of the active sharding strategy."""
        return self._strategy
