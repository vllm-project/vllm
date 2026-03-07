# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model executor layers.

For generic model support, users can use:
- TrainableFlashAttention: Training-compatible flash attention
- Megatron-LM parallel layers (via megatron_utils)
- vLLM's built-in parallel layers (via linear module)

Example with Megatron-LM:
    ```python
    # Install: pip install megatron-core
    from vllm.model_executor.layers.megatron_utils import (
        MegatronColumnParallelLinear,
        MegatronRowParallelLinear,
        create_megatron_config,
    )
    from vllm.model_executor.custom_models import TrainableFlashAttention


    class MyModel(nn.Module):
        def __init__(self, config):
            # Use Megatron's parallel layers
            megatron_config = create_megatron_config(tensor_model_parallel_size=4)
            self.fc = MegatronColumnParallelLinear(
                768, 3072, megatron_config=megatron_config
            )
            # Use vLLM's trainable attention
            self.attn = TrainableFlashAttention(768, 12)
    ```

Example with vLLM's built-in parallel layers:
    ```python
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.custom_models import TrainableFlashAttention
    ```
"""

# NOTE: TrainableFlashAttention is NOT imported here to avoid circular imports.
# Users should import it directly:
#   from vllm.model_executor.custom_models import TrainableFlashAttention

# Import vLLM's parallel layers for convenience
try:
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    _VLLM_PARALLEL_LAYERS_AVAILABLE = True
except ImportError:
    _VLLM_PARALLEL_LAYERS_AVAILABLE = False
    ColumnParallelLinear = None  # type: ignore
    RowParallelLinear = None  # type: ignore

__all__ = [
    # vLLM's parallel layer implementations
    "ColumnParallelLinear",
    "RowParallelLinear",
]
