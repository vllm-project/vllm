# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for issue #36478:
  - Bug A: IndexError in set_lora / slice_lora_b when len(lora_a/lora_b)
           < n_slices in certain TP configurations.
  - Bug B: Tensor size mismatch in MergedQKVParallelLinearWithLoRA when
           loading LoRA adapters on GQA models where Q output dim != KV
           output dim (e.g. Qwen3.5-2B: Q=4096, K=V=1024).

Run with:
    pytest tests/lora/test_layers_issue_36478.py -xvs
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_lora_config(max_lora_rank: int = 16, max_loras: int = 4):
    """Return a minimal LoRAConfig without importing the full vllm stack."""
    from vllm.config.lora import LoRAConfig

    return LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        lora_dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
# Bug A  –  IndexError in MergedColumnParallelLinearWithLoRA.slice_lora_b
#           and set_lora when len(lora_b) < n_slices
# ---------------------------------------------------------------------------


class TestBugA_BoundsChecking:
    """Tests for the IndexError fix (issue #36372 / #36478 Bug A)."""

    def test_slice_lora_b_short_list_no_index_error(self, default_vllm_config):
        """slice_lora_b must not raise IndexError when len(lora_b) < n_slices."""
        from vllm.lora.layers.column_parallel_linear import (
            MergedColumnParallelLinearWithLoRA,
        )
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        layer = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[4096, 4096],
            bias=False,
            params_dtype=torch.float16,
        )
        lora_config = _make_lora_config()
        lora_layer = MergedColumnParallelLinearWithLoRA(layer)
        lora_layer.create_lora_weights(4, lora_config)

        # Only 1 element even though n_slices == 2
        lora_b_short = [None]

        try:
            result = lora_layer.slice_lora_b(lora_b_short)
        except IndexError:
            pytest.fail("slice_lora_b raised IndexError when len(lora_b) < n_slices")

        assert len(result) == 2, f"Expected 2 slices, got {len(result)}"
        assert result == [None, None], f"Expected [None, None], got {result}"

    def test_set_lora_short_list_no_index_error(self, default_vllm_config):
        """set_lora must not raise IndexError when len(lora_a/lora_b) < n_slices."""
        from vllm.lora.layers.column_parallel_linear import (
            MergedColumnParallelLinearWithLoRA,
        )
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        layer = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[4096, 4096],
            bias=False,
            params_dtype=torch.float16,
        )
        lora_config = _make_lora_config()
        lora_layer = MergedColumnParallelLinearWithLoRA(layer)
        lora_layer.create_lora_weights(4, lora_config)

        # Only 1 element even though n_slices == 2
        lora_a_short = [torch.randn(16, 4096, dtype=torch.float16)]
        lora_b_short = [torch.randn(4096, 16, dtype=torch.float16)]

        try:
            lora_layer.set_lora(0, lora_a_short, lora_b_short)
        except IndexError:
            pytest.fail("set_lora raised IndexError when len(lora_a/lora_b) < n_slices")

        # Slice 0 should be set, slice 1 should remain zero
        assert torch.any(lora_layer.lora_a_stacked[0][0] != 0), (
            "lora_a slice 0 should be non-zero after set_lora"
        )
        assert torch.all(lora_layer.lora_a_stacked[1][0] == 0), (
            "lora_a slice 1 should remain zero (not provided)"
        )


# ---------------------------------------------------------------------------
# Bug B  –  Tensor size mismatch in MergedQKVParallelLinearWithLoRA
#           for GQA models (Qwen3.5-2B: Q=4096, K=V=1024)
# ---------------------------------------------------------------------------


class TestBugB_GQAShapeMismatch:
    """Tests for the GQA tensor size mismatch fix (issue #36478 Bug B)."""

    @pytest.mark.parametrize(
        "hidden_size, num_heads, num_kv_heads, head_dim, rank, description",
        [
            # Qwen3.5-2B style: 4:1 GQA ratio
            (2560, 32, 8, 128, 16, "Qwen3.5-2B (32Q / 8KV heads)"),
            # Llama-3.1-8B style: 8:1 GQA ratio
            (4096, 32, 8, 128, 16, "Llama3.1-8B (32Q / 8KV heads)"),
            # Mistral-7B style: 8:1 GQA ratio
            (4096, 32, 8, 128, 32, "Mistral-7B (32Q / 8KV heads)"),
            # MHA (no GQA) — must still work
            (4096, 32, 32, 128, 16, "MHA model (32Q / 32KV heads)"),
        ],
    )
    def test_create_lora_weights_gqa(
        self,
        default_vllm_config,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rank,
        description,
    ):
        """
        create_lora_weights must allocate lora_b_stacked with correct
        per-slice sizes for GQA models where Q output dim != KV output dim.
        """
        from vllm.lora.layers.column_parallel_linear import (
            MergedQKVParallelLinearWithLoRA,
        )
        from vllm.model_executor.layers.linear import QKVParallelLinear

        base_layer = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            params_dtype=torch.float16,
        )
        lora_config = _make_lora_config(max_lora_rank=rank)
        lora_layer = MergedQKVParallelLinearWithLoRA(base_layer)

        # Should not raise any shape errors
        lora_layer.create_lora_weights(4, lora_config)

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        assert lora_layer.lora_b_stacked[0].shape[2] == q_size, (
            f"[{description}] lora_b_stacked[0] (Q) should have size {q_size}, "
            f"got {lora_layer.lora_b_stacked[0].shape[2]}"
        )
        assert lora_layer.lora_b_stacked[1].shape[2] == kv_size, (
            f"[{description}] lora_b_stacked[1] (K) should have size {kv_size}, "
            f"got {lora_layer.lora_b_stacked[1].shape[2]}"
        )
        assert lora_layer.lora_b_stacked[2].shape[2] == kv_size, (
            f"[{description}] lora_b_stacked[2] (V) should have size {kv_size}, "
            f"got {lora_layer.lora_b_stacked[2].shape[2]}"
        )

    @pytest.mark.parametrize(
        "hidden_size, num_heads, num_kv_heads, head_dim, rank, description",
        [
            (2560, 32, 8, 128, 16, "Qwen3.5-2B (32Q / 8KV heads)"),
            (4096, 32, 8, 128, 16, "Llama3.1-8B (32Q / 8KV heads)"),
            (4096, 32, 32, 128, 16, "MHA model (32Q / 32KV heads)"),
        ],
    )
    def test_set_lora_gqa_no_size_mismatch(
        self,
        default_vllm_config,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rank,
        description,
    ):
        """
        set_lora must not raise 'size of tensor a must match size of tensor b'
        when loading separate q_proj / k_proj / v_proj LoRA weights on a GQA
        model (issue #36478).
        """
        from vllm.lora.layers.column_parallel_linear import (
            MergedQKVParallelLinearWithLoRA,
        )
        from vllm.model_executor.layers.linear import QKVParallelLinear

        base_layer = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            params_dtype=torch.float16,
        )
        lora_config = _make_lora_config(max_lora_rank=rank)
        lora_layer = MergedQKVParallelLinearWithLoRA(base_layer)
        lora_layer.create_lora_weights(4, lora_config)

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        # Simulate what PEFT produces: separate per-projection LoRA weights
        lora_a = [
            torch.randn(rank, hidden_size, dtype=torch.float16),  # q_proj A
            torch.randn(rank, hidden_size, dtype=torch.float16),  # k_proj A
            torch.randn(rank, hidden_size, dtype=torch.float16),  # v_proj A
        ]
        lora_b = [
            torch.randn(q_size, rank, dtype=torch.float16),  # q_proj B
            torch.randn(kv_size, rank, dtype=torch.float16),  # k_proj B
            torch.randn(kv_size, rank, dtype=torch.float16),  # v_proj B
        ]

        try:
            lora_layer.set_lora(0, lora_a, lora_b)
        except RuntimeError as e:
            if "size of tensor" in str(e):
                pytest.fail(
                    f"[{description}] set_lora raised tensor size mismatch "
                    f"(issue #36478): {e}"
                )
            raise

        # Verify the weights were actually written
        for i, name in enumerate(["Q", "K", "V"]):
            assert torch.any(lora_layer.lora_a_stacked[i][0] != 0), (
                f"lora_a stacked[{i}] ({name}) should be non-zero after set_lora"
            )
            assert torch.any(lora_layer.lora_b_stacked[i][0] != 0), (
                f"lora_b stacked[{i}] ({name}) should be non-zero after set_lora"
            )

    def test_set_lora_gqa_partial_lora_no_error(self, default_vllm_config):
        """
        set_lora with only q_proj LoRA (k/v are None) must not crash.
        This combines both Bug A and Bug B.
        """
        from vllm.lora.layers.column_parallel_linear import (
            MergedQKVParallelLinearWithLoRA,
        )
        from vllm.model_executor.layers.linear import QKVParallelLinear

        # Qwen3.5-2B style
        hidden_size, num_heads, num_kv_heads, head_dim = 2560, 32, 8, 128
        rank = 16
        q_size = num_heads * head_dim

        base_layer = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            params_dtype=torch.float16,
        )
        lora_config = _make_lora_config(max_lora_rank=rank)
        lora_layer = MergedQKVParallelLinearWithLoRA(base_layer)
        lora_layer.create_lora_weights(4, lora_config)

        # Only Q has a LoRA; K and V are None
        lora_a = [
            torch.randn(rank, hidden_size, dtype=torch.float16),
            None,
            None,
        ]
        lora_b = [
            torch.randn(q_size, rank, dtype=torch.float16),
            None,
            None,
        ]

        try:
            lora_layer.set_lora(0, lora_a, lora_b)
        except (IndexError, RuntimeError) as e:
            pytest.fail(f"set_lora raised an error with partial (Q-only) LoRA: {e}")

        assert torch.any(lora_layer.lora_a_stacked[0][0] != 0)
        assert torch.all(lora_layer.lora_b_stacked[1][0] == 0)
        assert torch.all(lora_layer.lora_b_stacked[2][0] == 0)
