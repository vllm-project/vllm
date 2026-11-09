# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GC shared ownership and changing-input CUDA Graph replay on SM90/SM100.

Run in an environment with the optional gc-moe dependencies installed.
"""

import json

import pytest
import torch

from vllm.config import CUDAGraphMode, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MegaMoEActivation,
)
from vllm.model_executor.layers.fused_moe.gc_moe import GcMoEExperts
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
    ),
    reason="GC MoE requires SM90 or SM100-family GPUs",
)


@pytest.mark.parametrize(
    "tokens,hidden,top_k,strided",
    [
        (0, 512, 1, False),
        (1, 5120, 6, False),
        (2, 5120, 6, True),
        (7, 5120, 6, False),
        (32, 5120, 6, False),
        (65, 512, 32, False),
        (512, 8192, 8, False),
        (2, 512, 6, "large_offset"),
    ],
)
def test_gc_preparation_matches_original_quantization(
    dist_init, tokens, hidden, top_k, strided
):
    """Fusion must preserve FP8 bytes/scales, routes and graph input updates."""
    import gc_moe
    import torch.distributed as dist

    from vllm.model_executor.layers.fused_moe.gc_moe import _prepare_gc_inputs
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(42)
    buffer = gc_moe.get_symm_buffer_for_gc_moe_sm90(
        dist.group.WORLD, 32, tokens + 2, top_k, hidden, 256, num_shared_experts=1
    )
    # Offset views also exercise alignment-sensitive Triton compilation.
    if strided == "large_offset":
        x = torch.empty_strided(
            (tokens, hidden),
            (2**31 + 16, 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        x.normal_()
    else:
        x = torch.randn(
            (tokens + 2, hidden + (16 if strided else 0)),
            device="cuda",
            dtype=torch.bfloat16,
        )[1 : tokens + 1, :hidden]
    magnitudes = torch.tensor(
        [1e-8, 1e-4, 1, 2.0**50, 2.0**-50, 2.0**100], device="cuda"
    )
    x.mul_(magnitudes[torch.arange(hidden, device="cuda") // 128 % 6])
    if tokens:
        x[0, :128] = 0
        x[0, 128:256] = torch.linspace(-448, 448, 128, device="cuda")
    step = 2 if strided else 1
    ids = torch.randint(
        0,
        32,
        (tokens + 2, top_k * step),
        device="cuda",
        dtype=torch.int32 if strided else torch.int64,
    )[1:-1, ::step]
    weights = torch.rand((tokens + 2, top_k * step), device="cuda")[1:-1, ::step]
    if tokens:
        ids[0, 0] = -1

    def check():
        for group, power_of_two, data, scales in (
            (128, False, buffer.x, buffer.x_sf),
            (32, True, buffer.shared_l1_acts, buffer.shared_l1_acts_sf),
        ):
            if tokens:
                q, sf = per_token_group_quant_fp8(
                    x, group, eps=1e-4, use_ue8m0=power_of_two
                )
                torch.testing.assert_close(
                    data[:tokens].view(torch.uint8),
                    q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(scales[:tokens], sf, rtol=0, atol=0)
            assert torch.count_nonzero(data[tokens:].view(torch.uint8)) == 0
            assert torch.count_nonzero(scales[tokens:]) == 0
        torch.testing.assert_close(
            buffer.topk_idx[:tokens], ids.to(torch.int64), rtol=0, atol=0
        )
        torch.testing.assert_close(
            buffer.topk_weights[:tokens], weights, rtol=0, atol=0
        )
        assert torch.count_nonzero(buffer.topk_idx[tokens:]) == 0
        assert torch.count_nonzero(buffer.topk_weights[tokens:]) == 0

    _prepare_gc_inputs(x, weights, ids, buffer)
    check()
    if tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _prepare_gc_inputs(x, weights, ids, buffer)
        for value in (0, 1):
            x.fill_(value)
            ids.add_(1).remainder_(32)
            weights.mul_(-1)
            graph.replay()
            check()
        del graph
    buffer.destroy()


@pytest.mark.parametrize(
    "tokens,hidden,strided",
    [
        (0, 512, False),
        (2, 5120, True),
        (65, 5120, False),
        (513, 512, False),
        (1025, 512, False),
    ],
)
def test_sm100_preparation_packs_and_swizzles_shared_scales(
    dist_init, tokens, hidden, strided
):
    """SM100 aliases input data but needs K32 packed and block-M-swizzled scales."""
    import gc_moe
    import torch.distributed as dist

    from vllm.model_executor.layers.fused_moe.gc_moe import _prepare_gc_inputs_sm100
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(42)
    buffer = gc_moe.get_symm_buffer_for_gc_moe(
        dist.group.WORLD, 32, tokens + 2, 6, hidden, 256, num_shared_experts=1
    )
    x = torch.randn(
        (tokens + 2, hidden + (16 if strided else 0)),
        dtype=torch.bfloat16,
        device="cuda",
    )[1 : tokens + 1, :hidden]
    step = 2 if strided else 1
    ids = torch.randint(0, 32, (tokens, 6 * step), device="cuda", dtype=torch.int32)[
        :, ::step
    ]
    weights = torch.rand((tokens, 6 * step), device="cuda")[:, ::step]
    if tokens:
        ids[0, 0] = -1
        x[0, :128] = 0
    bm = gc_moe.get_block_m_for_gc_moe(1, 32, buffer.num_max_tokens_per_rank, tokens, 6)
    rows = torch.arange(tokens, device="cuda")
    within = rows % bm
    sf_rows = (
        rows // bm * ((bm + 127) // 128 * 128)
        + within // 128 * 128
        + within % 32 * 4
        + within % 128 // 32
    )

    def check():
        assert buffer.x.data_ptr() == buffer.shared_l1_acts.data_ptr()
        if tokens:
            quantized, sf = per_token_group_quant_fp8(x, 32, eps=1e-4, use_ue8m0=True)
            packed = (sf.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)
            torch.testing.assert_close(
                buffer.x[:tokens].view(torch.uint8),
                quantized.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(buffer.x_sf[:tokens], packed, rtol=0, atol=0)
            expected_shared_sf = torch.zeros_like(buffer.shared_l1_acts_sf)
            expected_shared_sf[sf_rows] = packed
            torch.testing.assert_close(
                buffer.shared_l1_acts_sf, expected_shared_sf, rtol=0, atol=0
            )
        assert torch.count_nonzero(buffer.x[tokens:].view(torch.uint8)) == 0
        assert torch.count_nonzero(buffer.x_sf[tokens:]) == 0
        torch.testing.assert_close(buffer.topk_idx[:tokens], ids.long(), rtol=0, atol=0)
        torch.testing.assert_close(
            buffer.topk_weights[:tokens], weights, rtol=0, atol=0
        )

    _prepare_gc_inputs_sm100(x, weights, ids, buffer)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _prepare_gc_inputs_sm100(x, weights, ids, buffer)
    for value in (0, 1):
        x.fill_(value)
        ids.add_(1).remainder_(32)
        weights.mul_(-1)
        graph.replay()
        check()
    del graph
    buffer.destroy()


@pytest.mark.parametrize("grouped", [False, True])
def test_sm100_weight_scales_preserve_ue8m0_values(grouped):
    """Packing must be lossless, including UE8M0's subnormal FP32 endpoint."""
    from vllm.model_executor.layers.fused_moe.gc_moe import _pack_gc_sm100_weights

    exponents = torch.tensor([0, 126, 127, 254], dtype=torch.uint8, device="cuda")
    expected = exponents.expand((2, 128, 4) if grouped else (128, 4)).contiguous()
    data = torch.zeros(
        (2, 128, 64) if grouped else (128, 128),
        dtype=torch.int8 if grouped else torch.float8_e4m3fn,
        device="cuda",
    )
    for scales in (expected, expected.view(torch.float8_e8m0fnu).float()):
        _, packed = _pack_gc_sm100_weights((data, scales))
        torch.testing.assert_close(packed, expected.view(torch.int32), rtol=0, atol=0)
        assert packed.stride(-2) == 1
        assert packed.stride(-1) == 128


@pytest.mark.parametrize("scale", [0.0, -1.0, 0.75, float("inf"), float("nan")])
def test_sm100_rejects_shared_scales_that_would_change_weights(scale):
    from vllm.model_executor.layers.fused_moe.gc_moe import _pack_gc_sm100_weights

    data = torch.zeros((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.full((128, 4), scale, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="exact finite UE8M0"):
        _pack_gc_sm100_weights((data, scales))


@pytest.mark.parametrize(
    "counts,capacity,expected",
    [
        ([1, 0, 0, 0], 512, 1),
        ([0, 2, 1, 0], 512, 2),
        ([0, 0, 0, 0], 512, 1),
        ([8, 8, 8, 8], 512, 8),
        ([1, 400, 0, 17], 512, 400),
        ([128, 128], 64, 64),
        (None, 512, 512),
    ],
)
def test_gc_bound_covers_remote_and_padded_batches(counts, capacity, expected):
    """Small decode batches must not select the capacity-sized kernel geometry."""
    from vllm.model_executor.layers.fused_moe.gc_moe import _get_num_tokens_bound

    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        dp_metadata=DPMetadata(torch.tensor(counts, device="cpu"))
        if counts is not None
        else None,
    )
    with override_forward_context(context):
        assert _get_num_tokens_bound(capacity) == expected
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    with override_forward_context(context):
        assert _get_num_tokens_bound(capacity) == capacity
    with override_forward_context(None):
        assert _get_num_tokens_bound(capacity) == capacity


@pytest.mark.parametrize("scale_block", [32, 128])
@pytest.mark.parametrize("routed", [False, True])
def test_fused_shared_runs_once_with_empty_routes_and_graph_replay(
    dist_init, tmp_path, scale_block, routed
):
    """Check route ownership/weighting and a single shared add across graph replay."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 512,
                "intermediate_size": 256,
                "num_attention_heads": 8,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "max_position_embeddings": 128,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
        )
    )
    config.scheduler_config.max_num_batched_tokens = 128
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        shared = DeepseekV4MLP(
            512,
            256,
            "silu",
            swiglu_limit=10.0,
            quant_config=Fp8Config(weight_block_size=[scale_block, scale_block]),
            prefix="model.layers.0.mlp.shared_experts",
        )
        experts = GcMoEExperts(
            config,
            num_experts=2,
            num_local_experts=2,
            experts_start_idx=0,
            top_k=1,
            hidden_size=512,
            intermediate_size=256,
            activation=MegaMoEActivation(clamp=10.0),
            weight_format=MEGA_MOE_MXFP4,
            shared_experts=shared,
            prefix="model.layers.0.mlp.experts",
        )
        assert experts.has_fused_shared_experts
        for expert in range(2):
            for shard in ("w1", "w3", "w2"):
                prefix = "w2" if shard == "w2" else "w13"
                rows, cols = (512, 256) if shard == "w2" else (256, 512)
                for suffix, values in (
                    (
                        "weight",
                        torch.full(
                            (rows, cols // 2),
                            0x22 if routed and expert == 0 else 0,
                            dtype=torch.uint8,
                        ),
                    ),
                    (
                        "weight_scale",
                        torch.full((rows, cols // 32), 122, dtype=torch.uint8),
                    ),
                ):
                    name = f"{prefix}_{suffix}"
                    experts.weight_loader(
                        getattr(experts, name), values, name, shard, expert
                    )
        for shard in ("w1", "w3", "w2"):
            rows, cols = (512, 256) if shard == "w2" else (256, 512)
            base = f"model.layers.0.mlp.shared_experts.{shard}"
            experts.load_shared_expert_weight(
                f"{base}.weight",
                torch.full((rows, cols), 1 / 128, dtype=torch.float8_e4m3fn),
            )
            experts.load_shared_expert_weight(
                f"{base}.weight_scale",
                torch.full(
                    (rows // scale_block, cols // scale_block),
                    127 if scale_block == 32 else 1,
                    dtype=torch.uint8 if scale_block == 32 else torch.float32,
                ),
            )
        experts.process_weights_after_loading()
        x = torch.ones((2, 512), dtype=torch.bfloat16)
        ids = torch.arange(2, dtype=torch.int64).view(2, 1)
        weights = torch.full((2, 1), 0.5, dtype=torch.float32)
        result = experts(x, weights, ids)
        # Shared FC1 gives 4; K32 FP8 quantization rounds SiLU(4)*4 to 16;
        # shared FC2 gives 256*16/128 = 32. A duplicate shared add gives 64.
        expected = torch.full_like(x, 32)
        if routed:
            # Expert 0 has weights 1/32: FC1 clamps to 10, its weighted
            # SiLU product rounds to 50, and FC2 contributes 256*50/32 = 400.
            # Expert 1 is zero, so changing the selected expert is observable.
            expected[0] += 400
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        context = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            # Match the GPU runner's FULL capture entry point.
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            dp_metadata=DPMetadata(torch.tensor([2], device="cpu")),
        )
        with override_forward_context(context), torch.cuda.graph(graph):
            from vllm.model_executor.layers.fused_moe.gc_moe import (
                _get_num_tokens_bound,
            )

            assert _get_num_tokens_bound(128) == 2
            captured = experts(x, weights, ids)
        x.zero_()
        graph.replay()
        torch.testing.assert_close(captured, torch.zeros_like(x), rtol=0, atol=0)
        x.fill_(1)
        graph.replay()
        torch.testing.assert_close(captured, result, rtol=0, atol=0)
        context.dp_metadata = DPMetadata(torch.tensor([0], device="cpu"))
        with override_forward_context(context):
            empty = experts(x[:0], weights[:0], ids[:0])
        assert empty.shape == (0, 512)
