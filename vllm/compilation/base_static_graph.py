# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from typing import Any, Callable, Protocol

from vllm.config import VllmConfig


class AbstractStaticGraphWrapper(Protocol):
    """
    StaticGraphWrapper interface that allows platforms to wrap a callable
    to be captured as a static graph.
    """

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 graph_pool: Any, runtime_style: enum.Enum, **kwargs):
        """
        Initializes the StaticGraphWrapper class with graph capturing and
        execution-related configurations.

        Args:
            runnable (Callable): The callable to be wrapped and captured.
            vllm_config (VllmConfig): Global configuration for vLLM.
            graph_pool (Any):
                Graph memory pool handle, e.g.,
                    `torch.cuda.graph_pool_handle()`.
            runtime_style (enum.Enum): The style of the static
                graph runtime. e.g. see CUDAGraphRuntimeStyle in vllm/config.py.
        Keyword Args:
            kwargs: Additional keyword arguments for platform-specific
                configurations.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """
        Executes the wrapped callable.

        This may involve replaying a captured static graph if the conditions
        are met, or running the original callable eagerly and potentially
        capturing it.

        Args:
            *args: Variable length input arguments to be passed into the
                callable.
            **kwargs: Keyword arguments to be passed into the callable.

        Returns:
            Any: Output of the executed callable.
        """
        raise NotImplementedError
