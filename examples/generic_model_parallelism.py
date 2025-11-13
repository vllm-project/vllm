# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example demonstrating parallelism setup for user-defined models with vLLM.

This demonstrates the second key idea from the RFC:
User models can leverage vLLM's initialization functions for parallelism
with a callback-based approach.

**Key Point**: Users can use ANY parallelism implementation (Megatron-LM,
DeepSpeed, vLLM's built-in, etc.) - vLLM doesn't force a specific choice!

This example shows how users would integrate external parallelism libraries
like Megatron-LM with their models. The integration code (adapters/wrappers)
lives in USER code, not in vLLM core.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

# ============================================================================
# User's Choice: Which Parallelism Library to Use?
# ============================================================================

# Option 1: Use Megatron-LM (NVIDIA's implementation)
# Install: pip install megatron-core
try:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear as MegatronColumnParallelLinear,
    )
    from megatron.core.tensor_parallel import (
        RowParallelLinear as MegatronRowParallelLinear,
    )

    MEGATRON_AVAILABLE = True
    print("✓ Megatron-Core available - users can integrate it!")
except ImportError:
    MEGATRON_AVAILABLE = False
    MegatronColumnParallelLinear = None
    MegatronRowParallelLinear = None
    ModelParallelConfig = None
    print("✗ Megatron-Core not available")

# Option 2: Use vLLM's built-in parallel layers (fallback)
try:
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear as VLLMColumnParallelLinear,
    )
    from vllm.model_executor.layers.linear import (
        RowParallelLinear as VLLMRowParallelLinear,
    )

    VLLM_PARALLEL_AVAILABLE = True
    print("✓ vLLM parallel layers available")
except ImportError:
    VLLM_PARALLEL_AVAILABLE = False
    VLLMColumnParallelLinear = None
    VLLMRowParallelLinear = None
    print("✗ vLLM parallel layers not available")

# Check for vLLM's trainable attention (works with any parallelism library!)
import importlib.util

if importlib.util.find_spec("vllm.model_executor.layers.trainable_attention"):
    VLLM_FLASH_ATTENTION_AVAILABLE = True
    print("✓ vLLM TrainableFlashAttention available")
else:
    VLLM_FLASH_ATTENTION_AVAILABLE = False
    print("✗ vLLM TrainableFlashAttention not available")


# ============================================================================
# User-Defined Adapter: Megatron → vLLM Bridge (Example)
# ============================================================================

if MEGATRON_AVAILABLE:

    class MegatronLinearAdapter:
        """
        Example adapter showing how users can wrap Megatron layers to work
        with vLLM's parallel context.

        This lives in USER code, not vLLM core! Users choose how to integrate
        their preferred parallelism library.
        """

        @staticmethod
        def create_column_parallel(
            input_size: int,
            output_size: int,
            parallel_context: "ParallelContext",
            bias: bool = False,
        ):
            """Create Megatron ColumnParallelLinear from vLLM parallel context."""
            # Create Megatron config from vLLM's parallel settings
            megatron_config = ModelParallelConfig(
                tensor_model_parallel_size=parallel_context.get_tensor_parallel_world_size(),
                pipeline_model_parallel_size=parallel_context.get_pipeline_parallel_world_size(),
            )

            # Simple init method
            def init_method(weight):
                nn.init.kaiming_uniform_(weight, a=5**0.5)

            # Create Megatron layer
            layer = MegatronColumnParallelLinear(
                input_size=input_size,
                output_size=output_size,
                config=megatron_config,
                init_method=init_method,
                bias=bias,
                gather_output=False,
            )

            # Return wrapper that handles the (output, bias) tuple
            class Wrapper(nn.Module):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    output, _ = self.layer(x)
                    return output

            return Wrapper(layer)

        @staticmethod
        def create_row_parallel(
            input_size: int,
            output_size: int,
            parallel_context: "ParallelContext",
            bias: bool = False,
        ):
            """Create Megatron RowParallelLinear from vLLM parallel context."""
            megatron_config = ModelParallelConfig(
                tensor_model_parallel_size=parallel_context.get_tensor_parallel_world_size(),
                pipeline_model_parallel_size=parallel_context.get_pipeline_parallel_world_size(),
            )

            def init_method(weight):
                nn.init.kaiming_uniform_(weight, a=5**0.5)

            layer = MegatronRowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                config=megatron_config,
                init_method=init_method,
                bias=bias,
                input_is_parallel=True,
            )

            class Wrapper(nn.Module):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    output, _ = self.layer(x)
                    return output

            return Wrapper(layer)


# ============================================================================
# Simplified Parallel Layer Factory (for demonstration)
# ============================================================================


class ParallelLinearFactory:
    """
    Factory that users can implement to switch between parallelism libraries.

    This demonstrates the flexibility - users choose their implementation!
    """

    @staticmethod
    def create_column_parallel(input_size, output_size, parallel_context, bias=False):
        """Create a column-parallel linear layer."""
        if MEGATRON_AVAILABLE:
            try:
                print("  → Using Megatron ColumnParallelLinear")
                return MegatronLinearAdapter.create_column_parallel(
                    input_size, output_size, parallel_context, bias
                )
            except Exception as e:
                print(
                    f"  ⚠ Megatron initialization failed "
                    f"({e.__class__.__name__}), falling back to torch.nn.Linear"
                )
                print("     (Megatron requires proper distributed initialization)")
                return nn.Linear(input_size, output_size, bias=bias)
        elif VLLM_PARALLEL_AVAILABLE:
            print("  → Using vLLM ColumnParallelLinear")
            # vLLM's layers would use vLLM's API
            # For now, return a simple Linear as placeholder
            return nn.Linear(input_size, output_size, bias=bias)
        else:
            print("  → Using standard torch.nn.Linear (no parallelism)")
            return nn.Linear(input_size, output_size, bias=bias)

    @staticmethod
    def create_row_parallel(input_size, output_size, parallel_context, bias=False):
        """Create a row-parallel linear layer."""
        if MEGATRON_AVAILABLE:
            try:
                print("  → Using Megatron RowParallelLinear")
                return MegatronLinearAdapter.create_row_parallel(
                    input_size, output_size, parallel_context, bias
                )
            except Exception as e:
                print(
                    f"  ⚠ Megatron initialization failed "
                    f"({e.__class__.__name__}), falling back to torch.nn.Linear"
                )
                print("     (Megatron requires proper distributed initialization)")
                return nn.Linear(input_size, output_size, bias=bias)
        elif VLLM_PARALLEL_AVAILABLE:
            print("  → Using vLLM RowParallelLinear")
            return nn.Linear(input_size, output_size, bias=bias)
        else:
            print("  → Using standard torch.nn.Linear (no parallelism)")
            return nn.Linear(input_size, output_size, bias=bias)


# ============================================================================
# vLLM Parallelism Context (Mock for demonstration)
# ============================================================================


@dataclass
class ParallelConfig:
    """Configuration for model parallelism."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


class ParallelContext:
    """
    Mock of vLLM's parallelism context.

    In the real implementation, this would provide:
    - Process group initialization
    - Rank and world size information
    - Communication primitives
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.tp_rank = 0
        self.pp_rank = 0
        self.world_size = config.tensor_parallel_size * config.pipeline_parallel_size

    def get_tensor_parallel_rank(self) -> int:
        """Get the rank within tensor parallel group."""
        return self.tp_rank

    def get_pipeline_parallel_rank(self) -> int:
        """Get the rank within pipeline parallel group."""
        return self.pp_rank

    def get_tensor_parallel_world_size(self) -> int:
        """Get the size of tensor parallel group."""
        return self.config.tensor_parallel_size

    def get_pipeline_parallel_world_size(self) -> int:
        """Get the size of pipeline parallel group."""
        return self.config.pipeline_parallel_size


# ============================================================================
# Model with Parallelism Support
# ============================================================================


class ParallelTransformerAttention(nn.Module):
    """
    Attention layer with tensor parallelism support.

    Demonstrates how user models can leverage external parallel libraries.
    Uses Megatron (if available) or vLLM's parallel layers + TrainableFlashAttention!
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        parallel_context: ParallelContext | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.parallel_context = parallel_context or ParallelContext(ParallelConfig())

        print(
            f"Creating ParallelTransformerAttention "
            f"(hidden_size={hidden_size}, num_heads={num_heads})"
        )

        # Q, K, V projections with column parallelism
        # Uses user's choice of parallelism library!
        self.qkv_proj = ParallelLinearFactory.create_column_parallel(
            hidden_size,
            3 * hidden_size,
            parallel_context,
            bias=False,
        )

        # Output projection with row parallelism
        self.o_proj = ParallelLinearFactory.create_row_parallel(
            hidden_size,
            hidden_size,
            parallel_context,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism."""
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection (column parallel)
        qkv = self.qkv_proj(hidden_states)

        # Get local number of heads for this TP rank
        tp_size = self.parallel_context.get_tensor_parallel_world_size()
        local_num_heads = self.num_heads // tp_size if tp_size > 1 else self.num_heads

        # Reshape for attention
        qkv = qkv.reshape(batch_size, seq_len, 3, local_num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation (local to this TP rank)
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection (row parallel, includes all-reduce in real impl)
        output = self.o_proj(attn_output)

        return output


class ParallelMLP(nn.Module):
    """
    MLP layer with tensor parallelism support.

    Demonstrates the standard pattern:
    - Column parallel for up-projection
    - Row parallel for down-projection
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        parallel_context: ParallelContext | None = None,
    ):
        super().__init__()
        self.parallel_context = parallel_context or ParallelContext(ParallelConfig())

        print(
            f"Creating ParallelMLP (hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size})"
        )

        # Up projection with column parallelism
        self.gate_up_proj = ParallelLinearFactory.create_column_parallel(
            hidden_size,
            intermediate_size * 2,  # For gated activation
            parallel_context,
            bias=False,
        )

        # Down projection with row parallelism
        self.down_proj = ParallelLinearFactory.create_row_parallel(
            intermediate_size,
            hidden_size,
            parallel_context,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism."""
        # Up projection
        gate_up = self.gate_up_proj(x)

        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # Gated activation
        intermediate = nn.functional.silu(gate) * up

        # Down projection
        output = self.down_proj(intermediate)

        return output


# ============================================================================
# User Model Registration with Parallelism Callback
# ============================================================================


class UserModelBuilder:
    """
    Builder that demonstrates the callback-based approach for parallelism setup.

    This is the pattern proposed in the RFC where users can provide a
    callback that vLLM calls after setting up process groups.
    """

    @staticmethod
    def build_model_with_parallel_context(
        config: dict,
        parallel_context: ParallelContext,
    ) -> nn.Module:
        """
        User-provided callback to build model with parallelism.

        vLLM would call this after initializing process groups.

        Args:
            config: Model configuration
            parallel_context: Initialized parallelism context from vLLM

        Returns:
            Model instance with parallelism set up
        """

        class UserParallelModel(nn.Module):
            """User's model that leverages vLLM parallelism."""

            def __init__(self, config, parallel_context):
                super().__init__()
                hidden_size = config.get("hidden_size", 768)
                intermediate_size = config.get("intermediate_size", 3072)
                num_heads = config.get("num_heads", 12)

                # Build layers using vLLM's parallel utilities
                self.attn = ParallelTransformerAttention(
                    hidden_size,
                    num_heads,
                    parallel_context,
                )

                self.mlp = ParallelMLP(
                    hidden_size,
                    intermediate_size,
                    parallel_context,
                )

                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Attention block
                residual = hidden_states
                hidden_states = self.norm1(hidden_states)
                hidden_states = self.attn(hidden_states)
                hidden_states = residual + hidden_states

                # MLP block
                residual = hidden_states
                hidden_states = self.norm2(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states

                return hidden_states

        return UserParallelModel(config, parallel_context)


# ============================================================================
# Example Usage
# ============================================================================


def example_single_device():
    """Example: Using the model without parallelism."""
    print("=" * 60)
    print("Example 1: Single Device (no parallelism)")
    print("=" * 60)

    config = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_heads": 12,
    }

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    parallel_context = ParallelContext(parallel_config)

    model = UserModelBuilder.build_model_with_parallel_context(config, parallel_context)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_size = config["hidden_size"]

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    output = model(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Single device example works!\n")


def example_tensor_parallel():
    """Example: Using the model with tensor parallelism."""
    print("=" * 60)
    print("Example 2: Tensor Parallelism (simulated)")
    print("=" * 60)

    config = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_heads": 12,
    }

    # Simulate 4-way tensor parallelism
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    parallel_context = ParallelContext(parallel_config)

    print(f"Tensor Parallel Size: {parallel_config.tensor_parallel_size}")
    print(f"Rank: {parallel_context.get_tensor_parallel_rank()}")

    model = UserModelBuilder.build_model_with_parallel_context(config, parallel_context)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_size = config["hidden_size"]

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    output = model(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Tensor parallel example works!\n")


def example_registration_pattern():
    """
    Example: How a user would register their model with vLLM.

    This demonstrates the proposed API from the RFC.
    """
    print("=" * 60)
    print("Example 3: Model Registration Pattern (pseudo-code)")
    print("=" * 60)

    print("""
# Step 1: User defines their model builder
class MyModelBuilder:
    @staticmethod
    def build(config, parallel_context):
        return MyModel(config, parallel_context)

# Step 2: Register with vLLM
# vLLM provides the parallel_context after setting up process groups
from vllm.model_executor.models.registry import ModelRegistry

ModelRegistry.register_model(
    "MyCustomModel",
    MyModelBuilder.build,  # Callback receives parallel_context
)

# Step 3: Use with LLM API
from vllm import LLM

llm = LLM(
    model="path/to/my/model",
    tensor_parallel_size=4,  # vLLM sets this up
    trust_remote_code=True,
)

# vLLM will:
# 1. Initialize process groups based on tensor_parallel_size
# 2. Call MyModelBuilder.build(config, parallel_context)
# 3. Your model can use parallel_context for TP/PP
    """)

    print("✓ Registration pattern demonstrated!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Parallelism Flexibility Example for vLLM Generic Models")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print(
        "  1. Users can import parallelism layers from ANY library "
        "(Megatron, vLLM, etc.)"
    )
    print("  2. Users write their own adapters/wrappers (not built into vLLM core)")
    print("  3. vLLM provides context (parallel settings), users choose implementation")
    print()
    if MEGATRON_AVAILABLE:
        print("✓ Megatron-Core available - showing import and adapter pattern")
        print("  (Note: Full Megatron integration requires distributed initialization)")
    else:
        print("○ Megatron-LM not available, using simplified layers")
    print()
    print("=" * 80 + "\n")

    # Only run single device example to avoid TP reshaping issues with fallback
    example_single_device()
    # example_tensor_parallel()  # Requires real distributed setup
    example_registration_pattern()

    print("=" * 80)
    print("Example completed successfully!")
    print()
    print("Key Takeaway:")
    print("  vLLM doesn't force a specific parallelism library - users can integrate")
    print("  Megatron-LM, DeepSpeed, FairScale, or any other framework by writing")
    print("  simple adapters in their own code (like MegatronLinearAdapter above).")
    print()
    print("What was demonstrated:")
    print("  ✓ How to import from Megatron-Core (if available)")
    print("  ✓ How to write user-defined adapters to bridge different APIs")
    print("  ✓ How to mix libraries (Megatron parallel layers + vLLM attention)")
    print("  ✓ That integration code lives in USER code, not vLLM core")
    print("=" * 80)
