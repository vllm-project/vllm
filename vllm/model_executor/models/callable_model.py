# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model registry callback support for user-defined models.

This module provides support for registering models using factory functions/callbacks,
allowing users to receive vLLM's parallel context when building their models.

Example:
    ```python
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.parallel_context import ParallelContext


    def build_my_model(vllm_config, parallel_context: ParallelContext):
        # Use parallel context to configure your model
        tp_size = parallel_context.get_tensor_parallel_world_size()
        model = MyCustomModel(config=vllm_config.hf_config, tp_size=tp_size)
        return model


    ModelRegistry.register_model("MyModel", build_my_model)
    ```
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from vllm.model_executor.models.registry import _BaseRegisteredModel, _ModelInfo
from vllm.model_executor.parallel_context import ParallelContext


@dataclass(frozen=True)
class _CallableRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model registered via a factory function/callable.

    The callable signature must be:
        (vllm_config, parallel_context: ParallelContext) -> nn.Module
    """

    callable_factory: Callable[[Any, ParallelContext], nn.Module]
    model_arch: str

    def inspect_model_cls(self) -> _ModelInfo:
        """
        Return minimal model info for callable-based models.

        Since we don't have a class to inspect, we return default values.
        Users can override these by providing additional metadata.
        """
        return _ModelInfo(
            architecture=self.model_arch,
            is_text_generation_model=True,  # Default assumption
            is_pooling_model=False,
            default_pooling_type="LAST",
            supports_cross_encoding=False,
            supports_multimodal=False,
            supports_multimodal_raw_input_only=False,
            supports_multimodal_encoder_tp_data=False,
            supports_pp=False,  # Users can enable via their own implementation
            has_inner_state=False,
            is_attention_free=False,
            is_hybrid=False,
            has_noops=False,
            supports_mamba_prefix_caching=False,
            supports_transcription=False,
            supports_transcription_only=False,
        )

    def load_model_cls(self) -> type[nn.Module]:
        """
        Return a wrapper class that instantiates the model via the factory.

        The wrapper class will call the factory function with vllm_config
        and parallel_context in its __init__.
        """
        factory = self.callable_factory

        class CallableModelWrapper(nn.Module):
            """Wrapper that calls the factory function with parallel context."""

            def __init__(self, vllm_config=None, parallel_context=None, **kwargs):
                # Call super init FIRST
                super().__init__()

                # Extract parallel context from various possible sources
                if parallel_context is None:
                    # Try to create from vllm_config
                    if vllm_config is not None and hasattr(
                        vllm_config, "parallel_config"
                    ):
                        parallel_context = ParallelContext.from_config(
                            vllm_config.parallel_config
                        )
                    else:
                        # Default: no parallelism
                        parallel_context = ParallelContext(
                            tensor_model_parallel_size=1,
                            pipeline_model_parallel_size=1,
                            data_parallel_size=1,
                        )

                # Call the factory to create the actual model
                self._actual_model = factory(vllm_config, parallel_context)

                # Copy attributes from actual model to this wrapper
                # This ensures that vLLM can find the expected attributes
                if hasattr(self._actual_model, "config"):
                    self.config = self._actual_model.config

            def forward(self, *args, **kwargs):
                """Forward to the actual model."""
                return self._actual_model.forward(*args, **kwargs)

            def __getattr__(self, name):
                """Delegate attribute access to the actual model."""
                if name == "_actual_model":
                    return super().__getattr__(name)
                return getattr(self._actual_model, name)

        # Set the name to match the architecture
        CallableModelWrapper.__name__ = self.model_arch
        CallableModelWrapper.__qualname__ = self.model_arch

        return CallableModelWrapper


__all__ = ["_CallableRegisteredModel"]
