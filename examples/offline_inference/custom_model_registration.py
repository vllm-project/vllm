# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Registering a Custom Model with vLLM

This example demonstrates how to register a custom model with vLLM's model
registry using the callback registration system. This allows you to use your
own model architectures with vLLM for fast inference.

Key features:
- Minimal interface requirements (get_input_embeddings + forward)
- Access to vLLM's parallelism configuration (TP/PP/DP)
- Support for any callable (function, lambda, closure, callable object)
- No strict inheritance requirements

For more details, see the vLLM documentation on custom models.
"""

import torch
import torch.nn as nn

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


# ============================================================================
# Step 1: Define Your Model
# ============================================================================
class CustomModel(nn.Module):
    """
    A minimal custom model that satisfies vLLM's interface requirements.

    Required methods:
        - get_input_embeddings(input_ids) -> embeddings
        - forward(input_ids, positions, **kwargs) -> hidden_states

    Optional methods (vLLM provides defaults if not implemented):
        - compute_logits(hidden_states, sampling_metadata) -> logits
        - load_weights(weights) - for loading checkpoints
        - sample(logits, sampling_metadata) - for token sampling
    """

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Simple model components
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Model forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len] - position indices
            **kwargs: Additional vLLM arguments (can be ignored for simple models)

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
                          NOTE: Return hidden states, not logits.
                          vLLM calls compute_logits() separately.
        """
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """
        Compute output logits from hidden states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            sampling_metadata: vLLM sampling metadata (optional)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)


# ============================================================================
# Step 2: Create a Factory Function
# ============================================================================
def build_custom_model(vllm_config, parallel_context: ParallelContext):
    """
    Factory function that builds your model.

    This function is called by vLLM when loading your model. It receives:
        - vllm_config: vLLM's configuration (includes hf_config, etc.)
        - parallel_context: Parallelism info (TP/PP ranks and world sizes)

    Returns:
        Your model instance (nn.Module)
    """
    # Access parallelism information
    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()

    print(f"Building model on TP rank {tp_rank}/{tp_size}")

    # Extract model configuration
    if hasattr(vllm_config, "hf_config") and vllm_config.hf_config:
        vocab_size = getattr(vllm_config.hf_config, "vocab_size", 32000)
        hidden_size = getattr(vllm_config.hf_config, "hidden_size", 4096)
    else:
        vocab_size = 32000
        hidden_size = 4096

    # Build and return your model
    return CustomModel(vocab_size=vocab_size, hidden_size=hidden_size)


# ============================================================================
# Step 3: Register Your Model
# ============================================================================

# Register the model with vLLM's model registry
ModelRegistry.register_model("CustomModel", build_custom_model)


# ============================================================================
# Alternative Registration Methods
# ============================================================================

# You can also use other callable types:

# Method 1: Lambda (for simple cases)
ModelRegistry.register_model("CustomModelLambda", lambda cfg, ctx: CustomModel())


# Method 2: Closure (to capture configuration)
def make_model_factory(hidden_size: int = 4096):
    """Create a factory that captures configuration."""

    def factory(vllm_config, parallel_context):
        return CustomModel(hidden_size=hidden_size)

    return factory


ModelRegistry.register_model("CustomModelClosure", make_model_factory(8192))


# Method 3: Callable object (for complex initialization logic)
class ModelBuilder:
    """Callable object that builds models."""

    def __init__(self, model_variant: str = "default"):
        self.model_variant = model_variant

    def __call__(self, vllm_config, parallel_context):
        print(f"Building {self.model_variant} variant")
        return CustomModel()


ModelRegistry.register_model(
    "CustomModelCallable", ModelBuilder(model_variant="optimized")
)


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Custom Model Registration Example")
    print("=" * 70)

    # After registration, you can use your model with vLLM's LLM() API:
    #
    # from vllm import LLM
    #
    # llm = LLM(
    #     model="path/to/your/model",  # Path containing config.json
    #     tensor_parallel_size=2,       # Use 2 GPUs with tensor parallelism
    # )
    #
    # outputs = llm.generate(["Hello, world!"])
    # print(outputs[0].outputs[0].text)

    print("\n✓ Models registered successfully!")
    print("\nRegistered model architectures:")
    print("  - CustomModel (function factory)")
    print("  - CustomModelLambda (lambda factory)")
    print("  - CustomModelClosure (closure factory)")
    print("  - CustomModelCallable (callable object factory)")

    # Simple test: Load and instantiate one of the models
    print("\n" + "=" * 70)
    print("Testing Model Instantiation")
    print("=" * 70)

    # Load the model class
    model_cls = ModelRegistry._try_load_model_cls("CustomModel")
    print(f"\n✓ Loaded model class: {model_cls.__name__}")

    # Create mock config and parallel context
    mock_config = type("Config", (), {"hf_config": None, "parallel_config": None})()
    ctx = ParallelContext(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    # Instantiate the model
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)
    print(f"✓ Created model instance: {type(model).__name__}")

    # Test forward pass
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    print(f"\n✓ Running forward pass with batch_size={batch_size}, seq_len={seq_len}")
    hidden_states = model.forward(input_ids=input_ids, positions=positions)
    print(f"  Hidden states shape: {hidden_states.shape}")

    logits = model.compute_logits(hidden_states)
    print(f"  Logits shape: {logits.shape}")

    print("\n" + "=" * 70)
    print("✅ Success! Your custom model works with vLLM!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Save your model config to a directory")
    print("  2. Use with: LLM(model='path/to/model')")
    print("  3. Generate text with: llm.generate(['Your prompt'])")
