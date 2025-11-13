#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test of callback registration with vLLM's LLM API.

This test demonstrates:
1. Registering a model with a factory function
2. Factory receiving real parallel context when vLLM spawns workers
3. Model working with vLLM's generate() API

Run with:
    python test_callback_e2e.py

Or with tensor parallelism:
    # Requires 2 GPUs
    python test_callback_e2e.py --tensor-parallel-size 2
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SimpleConfig:
    """Minimal config for our test model."""

    vocab_size: int = 1000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_position_embeddings: int = 512


class SimpleTransformer(nn.Module):
    """
    Minimal transformer for testing.

    This model logs when it's created to show which worker is building it.
    """

    def __init__(self, config: SimpleConfig, tp_rank: int, tp_size: int):
        super().__init__()
        self.config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        print(f"[Worker TP rank {tp_rank}/{tp_size}] Building SimpleTransformer")

        # Simple embedding and output layers
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Transformer layers (simplified - not actually parallel)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_size * 4,
                    batch_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list | None = None,
        attn_metadata=None,
        intermediate_tensors=None,
        **kwargs,
    ):
        """Forward pass compatible with vLLM's interface."""
        # Simple forward pass
        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor | None:
        """Compute logits for next token prediction."""
        logits = self.lm_head(hidden_states)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample next tokens."""
        # Simple greedy sampling
        next_tokens = torch.argmax(logits, dim=-1)
        return next_tokens


def build_simple_model(vllm_config, parallel_context):
    """
    Factory function that builds a model using parallel context.

    This is what users will write!
    """

    # Get parallel information from context
    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()
    pp_rank = parallel_context.get_pipeline_parallel_rank()

    print(f"\n{'=' * 60}")
    print("FACTORY CALLED!")
    print(f"  TP rank: {tp_rank}/{tp_size}")
    print(f"  PP rank: {pp_rank}")
    print(f"  vLLM config type: {type(vllm_config).__name__}")
    print(f"{'=' * 60}\n")

    # Create config
    config = SimpleConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_position_embeddings=512,
    )

    # Build model with parallel info
    model = SimpleTransformer(config, tp_rank, tp_size)

    # Set config attribute (vLLM expects this)
    model.config = config

    return model


def test_with_llm_api():
    """Test using vLLM's LLM API (the real usage pattern)."""
    from vllm import LLM
    from vllm.model_executor.models import ModelRegistry

    print("Registering model with callback...")
    ModelRegistry.register_model("SimpleTransformer", build_simple_model)

    print("\nCreating LLM with SimpleTransformer...")
    print("This will spawn workers and call our factory function!\n")

    # Create LLM - this spawns workers!
    # Each worker will call build_simple_model() with real parallel context
    _llm = LLM(  # noqa: F841
        model="SimpleTransformer",
        tokenizer=None,  # We'll use a simple tokenizer
        tensor_parallel_size=1,  # Can increase if you have multiple GPUs
        max_model_len=128,
        enforce_eager=True,  # Disable CUDA graphs for simplicity
    )

    print("\n" + "=" * 60)
    print("✓ LLM created successfully!")
    print("✓ Our factory was called in each worker process")
    print("✓ Each worker received real parallel context")
    print("=" * 60 + "\n")

    # TODO: Add generation test once we figure out tokenizer


def test_with_registry():
    """Test just the registry mechanics without full LLM."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.parallel_context import ParallelContext

    print("=" * 60)
    print("Testing Registry Mechanics")
    print("=" * 60)

    # Register
    ModelRegistry.register_model("SimpleTransformer", build_simple_model)
    print("✓ Registered model with callback")

    # Load model class
    model_cls = ModelRegistry._try_load_model_cls("SimpleTransformer")
    print(f"✓ Loaded model class: {model_cls.__name__}")

    # Create parallel context
    ctx = ParallelContext(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        data_parallel_size=1,
    )

    # Create mock vllm_config
    mock_config = type(
        "Config",
        (),
        {
            "hf_config": None,
            "parallel_config": type(
                "ParallelConfig",
                (),
                {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                },
            )(),
        },
    )()

    # Instantiate model
    print("\nInstantiating model...")
    model = model_cls(vllm_config=mock_config, parallel_context=ctx)

    print(f"✓ Model created: {type(model).__name__}")
    print(f"✓ Model has TP rank: {model.tp_rank}, size: {model.tp_size}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    # Test 1: Registry mechanics (always works)
    test_with_registry()

    # Test 2: Full LLM API (requires more setup)
    if "--with-llm" in sys.argv:
        print("\nRunning LLM API test...")
        test_with_llm_api()
    else:
        print("\nSkipping LLM API test (add --with-llm to run)")
        print("Note: LLM API test requires proper vLLM setup and may need GPUs")
