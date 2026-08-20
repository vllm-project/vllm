# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import helion

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.register import register_kernel
from vllm.kernels.helion.utils import get_canonical_gpu_name

GPU_PLATFORM = get_canonical_gpu_name()

DEFAULT_CONFIGS: dict[CaseKey, helion.Config] = {
    CaseKey.default(): helion.Config(block_sizes=[32]),
}


@contextmanager
def dummy_kernel_registry(
    configs: dict[CaseKey, helion.Config] | None = None,
):
    """Context manager providing a register function with automatic config setup.

    Yields a ``register`` callable with the same signature as
    ``register_kernel``.  Before applying the real decorator it writes a
    config JSON for the kernel name (from ``op_name`` or ``fn.__name__``)
    into a temporary directory backed by a fresh ``ConfigManager``.
    """
    if configs is None:
        configs = DEFAULT_CONFIGS

    def _to_config_entries(cfgs: dict) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for k, v in cfgs.items():
            config_data = v.__dict__["config"]
            pairs.append({"key": dict(k), "config": config_data})
        return pairs

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        ConfigManager.reset_instance()
        cm = ConfigManager(base_dir=config_dir)

        with patch(
            "vllm.kernels.helion.config_manager.ConfigManager",
            return_value=cm,
        ):

            def register(
                op_name: str | None = None,
                **kwargs,
            ) -> Callable:
                def decorator(fn: Callable) -> Callable:
                    name = op_name or fn.__name__
                    kernel_dir = config_dir / name
                    kernel_dir.mkdir(parents=True, exist_ok=True)
                    (kernel_dir / f"{GPU_PLATFORM}.json").write_text(
                        json.dumps(_to_config_entries(configs))
                    )
                    return register_kernel(op_name, **kwargs)(fn)

                return decorator

            try:
                yield register
            finally:
                ConfigManager.reset_instance()
