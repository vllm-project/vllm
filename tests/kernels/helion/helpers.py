# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import helion

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.register import register_kernel
from vllm.kernels.helion.utils import get_canonical_gpu_name

GPU_PLATFORM = get_canonical_gpu_name()

DEFAULT_CONFIGS: dict[str, helion.Config] = {
    "default": helion.Config(block_sizes=[32]),
}


@contextmanager
def dummy_kernel_registry(
    configs: dict[str, helion.Config] | None = None,
):
    """Context manager providing a register function with automatic config setup.

    Yields a ``register`` callable with the same signature as
    ``register_kernel``.  Before applying the real decorator it writes a
    config JSON for the kernel name (from ``op_name`` or ``fn.__name__``)
    into a temporary directory backed by a fresh ``ConfigManager``.
    """
    if configs is None:
        configs = DEFAULT_CONFIGS
    config_data = {k: v.__dict__["config"] for k, v in configs.items()}

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
                        json.dumps(config_data)
                    )
                    return register_kernel(op_name, **kwargs)(fn)

                return decorator

            try:
                yield register
            finally:
                ConfigManager.reset_instance()
