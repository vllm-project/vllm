# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for generic model support RFC.

This test demonstrates two key concepts:
1. vLLM-provided flash attention module with backward pass for training
2. Parallelism setup allowing user models to leverage vLLM's initialization

The test creates a simple trainable transformer model in vanilla PyTorch,
registers it with vLLM, and demonstrates both inference and training.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

# vLLM imports - only import when needed to avoid initialization issues
# from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

# from vllm.config import ModelConfig


# ============================================================================
# Part 1: vLLM-provided Flash Attention with Backward Pass
# ============================================================================


class VLLMFlashAttention(nn.Module):
    """
    vLLM-provided flash attention module that supports both forward and backward.

    This wraps the existing vLLM flash attention forward pass and adds
    a custom backward pass for training scenarios.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout

        # Q, K, V projections
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_flash: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional flash attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            use_flash: Whether to use flash attention (if available and in training)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use flash attention if available and requested
        if use_flash and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # PyTorch 2.0+ flash attention
            # This is where vLLM's custom flash attention would be called
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(attention_mask is None),
            )
        else:
            # Standard attention fallback
            scale = self.head_dim**-0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.softmax(attn_weights, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)

            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


# ============================================================================
# Part 2: Simple Transformer Model with Parallelism Support
# ============================================================================


@dataclass
class GenericModelConfig:
    """Configuration for our generic model."""

    vocab_size: int = 1024
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_position_embeddings: int = 512
    dropout: float = 0.1

    # Parallelism configs (for future use with vLLM's parallel utils)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


class TransformerBlock(nn.Module):
    """A single transformer block using vLLM flash attention."""

    def __init__(self, config: GenericModelConfig):
        super().__init__()
        self.config = config

        # Use vLLM's flash attention
        self.self_attn = VLLMFlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Layer norms
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GenericTransformerForCausalLM(nn.Module):
    """
    A simple transformer model for demonstration.

    This model can be trained with standard PyTorch and then registered
    with vLLM for fast inference.
    """

    def __init__(self, config: GenericModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token indices
            position_ids: Optional [batch, seq_len] position indices

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


# ============================================================================
# Part 3: vLLM Integration Wrapper
# ============================================================================


class GenericTransformerForCausalLMVLLM(nn.Module):
    """
    Wrapper to make our model compatible with vLLM's interface.

    This demonstrates the "specification" part of the RFC - the minimum
    interface needed to work with vLLM.

    Implements VllmModelForTextGeneration interface:
    - __init__(vllm_config, prefix="")
    - get_input_embeddings(input_ids)
    - forward(input_ids, positions)
    - compute_logits(hidden_states)
    """

    # vLLM interface flags
    supports_pp = False  # Pipeline parallelism
    supports_multimodal = False

    def __init__(self, vllm_config=None, prefix: str = "", **kwargs):
        super().__init__()

        # Handle both vllm_config and legacy config args
        config = vllm_config or kwargs.get("config")

        # Convert vLLM config to our config
        if config is not None and hasattr(config, "hf_config"):
            # Extract from HuggingFace config
            hf_config = config.hf_config
            model_config = GenericModelConfig(
                vocab_size=getattr(hf_config, "vocab_size", 1024),
                hidden_size=getattr(hf_config, "hidden_size", 256),
                num_layers=getattr(hf_config, "num_hidden_layers", 4),
                num_heads=getattr(hf_config, "num_attention_heads", 4),
                max_position_embeddings=getattr(
                    hf_config, "max_position_embeddings", 512
                ),
            )
        else:
            # Use defaults
            model_config = GenericModelConfig()

        self.model = GenericTransformerForCausalLM(model_config)
        self.config = model_config

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply token embeddings to input_ids.

        Required by VllmModel interface.
        """
        return self.model.embed_tokens(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Required signature: forward(input_ids, positions, ...)

        Returns:
            hidden_states (not logits) - logits are computed by compute_logits()
        """
        # Get position embeddings
        if positions is None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)
        position_embeds = self.model.embed_positions(positions)
        hidden_states = inputs_embeds + position_embeds

        # Transformer blocks
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)

        # Final norm (but don't compute logits yet)
        hidden_states = self.model.norm(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Required by VllmModelForTextGeneration interface.

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        logits = self.model.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        """vLLM weight loading interface."""
        # In a real implementation, this would load weights from the weights iterator
        # For now, we'll just initialize randomly
        pass

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """vLLM sampling interface."""
        # This would integrate with vLLM's sampling logic
        # For now, just return the logits
        return logits


# ============================================================================
# Tests
# ============================================================================


class TestGenericModelSupport:
    """Test suite for generic model support."""

    def test_flash_attention_forward(self):
        """Test that flash attention forward pass works."""
        batch_size = 2
        seq_len = 16
        hidden_size = 256
        num_heads = 4

        attn = VLLMFlashAttention(hidden_size, num_heads)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_flash_attention_backward(self):
        """Test that flash attention backward pass works for training."""
        batch_size = 2
        seq_len = 16
        hidden_size = 256
        num_heads = 4

        attn = VLLMFlashAttention(hidden_size, num_heads)
        attn.train()

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, requires_grad=True
        )

        output = attn(hidden_states)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert hidden_states.grad is not None
        assert attn.qkv.weight.grad is not None
        assert attn.o_proj.weight.grad is not None

    def test_model_forward(self):
        """Test that the full model forward pass works."""
        config = GenericModelConfig(
            vocab_size=1024,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
        )

        model = GenericTransformerForCausalLM(config)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_model_training(self):
        """Test that the model can be trained."""
        config = GenericModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
        )

        model = GenericTransformerForCausalLM(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simple training step
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            target_ids.reshape(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_vllm_wrapper(self):
        """Test the vLLM wrapper interface."""
        # Create a mock config
        config = type(
            "Config",
            (),
            {
                "hf_config": type(
                    "HFConfig",
                    (),
                    {
                        "vocab_size": 1024,
                        "hidden_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "max_position_embeddings": 512,
                    },
                )()
            },
        )()

        model = GenericTransformerForCausalLMVLLM(vllm_config=config)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1024, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Test forward (returns hidden states)
        hidden_states = model(input_ids, positions)
        assert hidden_states.shape == (batch_size, seq_len, 256)

        # Test compute_logits (returns logits)
        logits = model.compute_logits(hidden_states)
        assert logits.shape == (batch_size, seq_len, 1024)

        # Test get_input_embeddings
        embeddings = model.get_input_embeddings(input_ids)
        assert embeddings.shape == (batch_size, seq_len, 256)

    @pytest.mark.skip(reason="Requires full vLLM setup with model files")
    def test_model_registration(self):
        """
        Test model registration with vLLM.

        This demonstrates how a user would register their model.
        """
        from vllm.model_executor.models.registry import ModelRegistry

        # Register the model
        ModelRegistry.register_model(
            "GenericTransformerForCausalLM",
            GenericTransformerForCausalLMVLLM,
        )

        # Verify it's registered
        assert "GenericTransformerForCausalLM" in ModelRegistry.get_supported_archs()

    @pytest.mark.skip(reason="Requires full vLLM setup with model files and GPU")
    def test_llm_api_integration(self):
        """
        Test integration with vLLM's LLM() API.

        This would be the end-to-end flow:
        1. Train model with PyTorch
        2. Save to disk
        3. Register with vLLM
        4. Load with LLM() API for fast inference
        """
        # Save model
        config = GenericModelConfig()
        _model = GenericTransformerForCausalLM(config)  # noqa: F841

        # In practice, you'd save this to disk in HF format
        # torch.save(_model.state_dict(), "model.pt")

        # Register
        ModelRegistry.register_model(
            "GenericTransformerForCausalLM",
            GenericTransformerForCausalLMVLLM,
        )

        # Load with vLLM (would need actual model files)
        # llm = LLM(
        #     model="path/to/model",
        #     trust_remote_code=True,
        # )

        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # outputs = llm.generate(["Hello, my name is"], sampling_params)

        pass


if __name__ == "__main__":
    # Run basic tests
    print("Testing Flash Attention Forward...")
    test = TestGenericModelSupport()
    test.test_flash_attention_forward()
    print("✓ Flash Attention Forward works")

    print("\nTesting Flash Attention Backward...")
    test.test_flash_attention_backward()
    print("✓ Flash Attention Backward works")

    print("\nTesting Model Forward...")
    test.test_model_forward()
    print("✓ Model Forward works")

    print("\nTesting Model Training...")
    test.test_model_training()
    print("✓ Model Training works")

    print("\nTesting vLLM Wrapper...")
    test.test_vllm_wrapper()
    print("✓ vLLM Wrapper works")

    print("\n" + "=" * 50)
    print("All basic tests passed!")
    print("=" * 50)
