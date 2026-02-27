# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FusedMoE.weight_loader MXFP4 per-expert 2-D loading path.

Validates the fix for https://github.com/vllm-project/vllm/issues/35324
where the MXFP4 branch crashed on standard HuggingFace MoE checkpoints
that store one 2-D tensor [out, in] per expert (instead of the gpt-oss
combined 3-D format [num_experts, out, in]).
"""

import pytest
import torch


def round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _build_mxfp4_params(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    tp_size: int,
    pad_align: int = 128,
):
    """Build param tensors matching Mxfp4MoEMethod.create_weights() shapes."""
    inter_per_tp = intermediate_size // tp_size
    inter_padded = round_up(inter_per_tp, pad_align)
    mxfp4_block = 32

    w13_weight = torch.zeros(
        num_experts, 2 * inter_padded, hidden_size // 2, dtype=torch.uint8
    )
    w2_weight = torch.zeros(
        num_experts, hidden_size, inter_padded // 2, dtype=torch.uint8
    )
    w13_weight_scale = torch.zeros(
        num_experts,
        2 * inter_padded,
        hidden_size // mxfp4_block,
        dtype=torch.uint8,
    )
    w2_weight_scale = torch.zeros(
        num_experts,
        hidden_size,
        inter_padded // mxfp4_block,
        dtype=torch.uint8,
    )
    w13_bias = torch.zeros(num_experts, 2 * inter_padded, dtype=torch.bfloat16)
    w2_bias = torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16)

    return {
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_weight_scale": w13_weight_scale,
        "w2_weight_scale": w2_weight_scale,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
        "inter_padded": inter_padded,
    }


def _simulate_mxfp4_weight_loader(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    tp_size: int,
    tp_rank: int,
):
    """Simulate the MXFP4 per-expert 2-D branch of weight_loader."""
    # Dtype guard (mirrors production weight_loader).
    if "bias" not in weight_name and loaded_weight.dtype != torch.uint8:
        raise ValueError(
            f"MXFP4 quantization expects pre-quantized uint8 "
            f"weights, but got dtype={loaded_weight.dtype} for "
            f"'{weight_name}'."
        )

    expert_data = param[expert_id]
    shard_dim = 0 if shard_id in ("w1", "w3") else 1

    # TP-shard (w2 bias is row-parallel, not sharded).
    if not (shard_id == "w2" and "bias" in weight_name):
        shard_size = loaded_weight.shape[shard_dim] // tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )

    # Select destination within the merged w13 parameter.
    if shard_id in ("w1", "w3"):
        half = expert_data.shape[0] // 2
        offset = 0 if shard_id == "w1" else half
        dst = expert_data.narrow(0, offset, loaded_weight.shape[0])
    else:
        dst = expert_data

    # Padding-aware copy.
    if loaded_weight.dim() == 1:
        dst[: loaded_weight.shape[0]].copy_(loaded_weight)
    else:
        dst[: loaded_weight.shape[0], : loaded_weight.shape[1]].copy_(loaded_weight)


class TestMxfp4PerExpertWeightLoader:
    """Test MXFP4 weight_loader with per-expert 2-D weights."""

    @pytest.mark.parametrize(
        "intermediate_size,hidden_size",
        [
            (14336, 4096),  # Mixtral-like (no padding at pad_align=128)
            (1024, 2048),  # Small model (no padding)
        ],
    )
    def test_per_expert_2d_tp1(self, intermediate_size, hidden_size):
        """TP=1: basic per-expert loading without TP sharding.

        Regression test for Bug 1 (IndexError on 2-D weights).
        """
        num_experts = 4
        tp_size, tp_rank = 1, 0
        params = _build_mxfp4_params(
            num_experts, intermediate_size, hidden_size, tp_size
        )

        # Create per-expert checkpoint weights (full, unsharded)
        w1_ckpt = torch.randint(
            0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
        )
        w3_ckpt = torch.randint(
            0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
        )
        w2_ckpt = torch.randint(
            0, 255, (hidden_size, intermediate_size // 2), dtype=torch.uint8
        )

        eid = 1
        for shard_id, ckpt, param_name in [
            ("w1", w1_ckpt, "w13_weight"),
            ("w3", w3_ckpt, "w13_weight"),
            ("w2", w2_ckpt, "w2_weight"),
        ]:
            _simulate_mxfp4_weight_loader(
                params[param_name],
                ckpt,
                param_name,
                shard_id,
                eid,
                tp_size,
                tp_rank,
            )

        # Verify w1 in first half of w13
        inter_padded = params["inter_padded"]
        assert torch.equal(
            params["w13_weight"][eid, :intermediate_size, : hidden_size // 2],
            w1_ckpt,
        )
        # Verify w3 in second half of w13
        assert torch.equal(
            params["w13_weight"][
                eid,
                inter_padded : inter_padded + intermediate_size,
                : hidden_size // 2,
            ],
            w3_ckpt,
        )
        # Verify w2
        assert torch.equal(
            params["w2_weight"][eid, :hidden_size, : intermediate_size // 2],
            w2_ckpt,
        )

    @pytest.mark.parametrize(
        "intermediate_size,hidden_size,pad_align",
        [
            (1408, 2048, 128),  # Qwen3-30B-A3B-like (704 -> 768 padding)
            (14336, 4096, 128),  # Mixtral-like (no padding)
        ],
    )
    def test_per_expert_2d_tp2_with_padding(
        self, intermediate_size, hidden_size, pad_align
    ):
        """TP=2: TP sharding + padding.

        Regression test for Bug 2 (TP shard size mismatch).
        """
        num_experts = 4
        tp_size = 2

        for tp_rank in range(tp_size):
            params = _build_mxfp4_params(
                num_experts, intermediate_size, hidden_size, tp_size, pad_align
            )

            w1_ckpt = torch.randint(
                0,
                255,
                (intermediate_size, hidden_size // 2),
                dtype=torch.uint8,
            )
            w3_ckpt = torch.randint(
                0,
                255,
                (intermediate_size, hidden_size // 2),
                dtype=torch.uint8,
            )
            w2_ckpt = torch.randint(
                0,
                255,
                (hidden_size, intermediate_size // 2),
                dtype=torch.uint8,
            )

            eid = 2
            for shard_id, ckpt, param_name in [
                ("w1", w1_ckpt, "w13_weight"),
                ("w3", w3_ckpt, "w13_weight"),
                ("w2", w2_ckpt, "w2_weight"),
            ]:
                _simulate_mxfp4_weight_loader(
                    params[param_name],
                    ckpt,
                    param_name,
                    shard_id,
                    eid,
                    tp_size,
                    tp_rank,
                )

            shard = intermediate_size // tp_size
            shard_w2 = (intermediate_size // 2) // tp_size
            inter_padded = params["inter_padded"]

            # Verify w1 TP shard in first half of w13
            assert torch.equal(
                params["w13_weight"][eid, :shard, : hidden_size // 2],
                w1_ckpt[shard * tp_rank : shard * (tp_rank + 1), :],
            )
            # Verify w3 TP shard in second half of w13
            assert torch.equal(
                params["w13_weight"][
                    eid,
                    inter_padded : inter_padded + shard,
                    : hidden_size // 2,
                ],
                w3_ckpt[shard * tp_rank : shard * (tp_rank + 1), :],
            )
            # Verify w2 TP shard
            assert torch.equal(
                params["w2_weight"][eid, :hidden_size, :shard_w2],
                w2_ckpt[:, shard_w2 * tp_rank : shard_w2 * (tp_rank + 1)],
            )

            # Verify padding regions are zeros
            if inter_padded > shard:
                assert params["w13_weight"][eid, shard:inter_padded, :].sum() == 0, (
                    "w1 padding region is not zero"
                )
                assert (
                    params["w13_weight"][eid, inter_padded + shard :, :].sum() == 0
                ), "w3 padding region is not zero"

    def test_per_expert_2d_bias(self):
        """Bias loading: w1/w3 column-parallel sharded, w2 unsharded."""
        num_experts = 4
        intermediate_size, hidden_size = 1408, 2048
        tp_size = 2

        for tp_rank in range(tp_size):
            params = _build_mxfp4_params(
                num_experts, intermediate_size, hidden_size, tp_size
            )

            w1_bias = torch.randn(intermediate_size, dtype=torch.bfloat16)
            w3_bias = torch.randn(intermediate_size, dtype=torch.bfloat16)
            w2_bias = torch.randn(hidden_size, dtype=torch.bfloat16)

            eid = 0
            for shard_id, ckpt, param_name in [
                ("w1", w1_bias, "w13_bias"),
                ("w3", w3_bias, "w13_bias"),
                ("w2", w2_bias, "w2_bias"),
            ]:
                _simulate_mxfp4_weight_loader(
                    params[param_name],
                    ckpt,
                    param_name,
                    shard_id,
                    eid,
                    tp_size,
                    tp_rank,
                )

            shard = intermediate_size // tp_size
            inter_padded = params["inter_padded"]

            # w1 bias: TP-sharded in first half
            assert torch.equal(
                params["w13_bias"][eid, :shard],
                w1_bias[shard * tp_rank : shard * (tp_rank + 1)],
            )
            # w3 bias: TP-sharded in second half
            assert torch.equal(
                params["w13_bias"][eid, inter_padded : inter_padded + shard],
                w3_bias[shard * tp_rank : shard * (tp_rank + 1)],
            )
            # w2 bias: NOT TP-sharded (full hidden_size)
            assert torch.equal(params["w2_bias"][eid, :hidden_size], w2_bias)

    def test_combined_3d_backward_compat(self):
        """3-D combined format (gpt-oss) still works with padding."""
        num_experts = 4
        param_3d = torch.zeros(num_experts, 2048, 1024, dtype=torch.uint8)
        loaded_3d = torch.randint(0, 255, (num_experts, 1536, 900), dtype=torch.uint8)

        # Simulate the 3-D combined branch
        param_3d[:, : loaded_3d.shape[1], : loaded_3d.shape[2]].copy_(loaded_3d)

        assert torch.equal(param_3d[:, :1536, :900], loaded_3d)
        # Padding region should remain zero
        assert param_3d[:, 1536:, :].sum() == 0
        assert param_3d[:, :, 900:].sum() == 0

    def test_combined_3d_bias_backward_compat(self):
        """3-D combined bias format (gpt-oss) still works."""
        num_experts = 4
        param_bias = torch.zeros(num_experts, 2048, dtype=torch.bfloat16)
        loaded_bias = torch.randn(num_experts, 1536, dtype=torch.bfloat16)

        # Simulate the 3-D combined bias branch
        param_bias[:, : loaded_bias.shape[1]].copy_(loaded_bias)

        assert torch.equal(param_bias[:, :1536], loaded_bias)
        assert param_bias[:, 1536:].sum() == 0

    def test_bf16_weight_rejected(self):
        """Loading BF16 weights into MXFP4 params raises ValueError.

        Regression test: prevents silent data corruption when a user
        passes --quantization mxfp4 with an unquantized BF16 checkpoint.
        """
        num_experts = 4
        intermediate_size, hidden_size = 1024, 2048
        tp_size, tp_rank = 1, 0
        params = _build_mxfp4_params(
            num_experts, intermediate_size, hidden_size, tp_size
        )

        # BF16 checkpoint weight (NOT pre-quantized)
        w1_ckpt_bf16 = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="MXFP4 quantization expects"):
            _simulate_mxfp4_weight_loader(
                params["w13_weight"],
                w1_ckpt_bf16,
                "w13_weight",
                "w1",
                expert_id=0,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

    def test_bf16_bias_accepted(self):
        """BF16 bias loading should still work (bias is always BF16)."""
        num_experts = 4
        intermediate_size, hidden_size = 1024, 2048
        tp_size, tp_rank = 1, 0
        params = _build_mxfp4_params(
            num_experts, intermediate_size, hidden_size, tp_size
        )

        w1_bias = torch.randn(intermediate_size, dtype=torch.bfloat16)

        # Should NOT raise — bias is BF16 in both checkpoint and params
        _simulate_mxfp4_weight_loader(
            params["w13_bias"],
            w1_bias,
            "w13_bias",
            "w1",
            expert_id=0,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
