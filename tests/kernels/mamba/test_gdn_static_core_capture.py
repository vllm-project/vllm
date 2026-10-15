# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable CUDA graph capture of the static GDN core.

The breakable CUDA graph of a token count captures the GDN core when the
metadata builder stages its capture buffers, and replays it for every batch
padded to that count. One graph, captured once, must equal the eager
``_forward_core`` bitwise for batches of every kind: the core output and every
conv and SSM state block except the null block, which pad rows may write.
"""

from __future__ import annotations

import types
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode, SpeculativeConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
    _qwen_gdn_attention_core_fused_norm_packed_is_capturable,
    _qwen_gdn_attention_core_is_capturable,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

H = 4  # key heads
HV = 8  # value heads
K = 128  # head_k_dim
V = 128  # head_v_dim
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
NUM_SPEC = 2
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
# Capacity of the builder's buffers, and the token count of the graph.
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 128
NUM_CAPTURED_TOKENS = 64
NUM_STATE_BLOCKS = 64

# (seq_lens, query_lens, num_decode_draft_tokens, is_prefilling), decode rows
# first as the scheduler orders them; None leaves a field unset.
BATCHES = {
    "spec_only": ([20, 30, 9], [3, 3, 3], [2, 2, 2], None),
    "prefill_only": ([40, 57], [40, 17], [-1, -1], None),
    "spec_and_prefill": ([20, 65, 100], [3, 1, 50], [2, -1, -1], None),
    "decode_and_prefill": ([40, 41, 200], [1, 1, 60], None, None),
    "one_token_prefills": ([40, 1, 1], [1, 1, 1], None, [False, True, True]),
    "spec_and_one_token_prefills": ([30, 1, 1], [3, 1, 1], [2, -1, -1], None),
    "max_sequences": (
        [30] * 6 + [10] * 10,
        [3] * 6 + [2] * 10,
        [2] * 6 + [-1] * 10,
        None,
    ),
    "all_tokens": ([30, 90], [3, 61], [2, -1], None),
    "zero_length_rows": ([40, 100, 16, 16], [1, 50, 0, 0], None, None),
    "spec_and_zero_length_rows": ([65, 20, 16], [1, 3, 0], [-1, 2, -1], None),
    # Eager runs causal_conv1d_update on a batch of only non-spec decodes,
    # the captured core causal_conv1d_fn, so only this one may differ.
    "decode_only": ([40, 30, 20], [1, 1, 1], None, None),
}
NOT_BITWISE_BATCHES = {"decode_only"}
# Tolerance of NOT_BITWISE_BATCHES: the conv rounding reaches the state.
NOT_BITWISE_TOLERANCE = 6e-2


def _make_builder(device: torch.device) -> GDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        hf_config_override={"linear_key_head_dim": K},
    )
    vllm_config.additional_config = {"gdn_prefill_backend": "triton"}
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.cache_config.mamba_cache_mode = "none"
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    # Breakable PIECEWISE graphs on MRV2 enable it; the test needs only the
    # buffers.
    builder.captures_mixed_batches = True
    return builder


def _build(
    builder: GDNAttentionMetadataBuilder, name: str, device: torch.device
) -> GDNAttentionMetadata:
    seq_lens, query_lens, draft_tokens, is_prefilling = BATCHES[name]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    # A distinct state block per row and spec token; block 0 is the null block.
    block_table = 1 + torch.randperm(NUM_STATE_BLOCKS - 1, device=device)[
        : batch.batch_size * (NUM_SPEC + 1)
    ].view(batch.batch_size, NUM_SPEC + 1).to(torch.int32)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, device).replace(
        block_table_tensor=block_table
    )
    if is_prefilling is not None:
        common = common.replace(is_prefilling=torch.tensor(is_prefilling))
    kwargs = {}
    if draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.randint(
            1, NUM_SPEC + 2, (batch.batch_size,), dtype=torch.int32, device=device
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


def _build_layer(
    vllm_config, kv_cache, A_log, dt_bias, conv_weight, conv_bias, norm_weight
):
    """A minimal object that runs the real ``_forward_core`` and
    ``_forward_core_fused_norm`` bound to it."""
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=conv_bias),
        kv_cache=kv_cache,
        norm=types.SimpleNamespace(
            weight=norm_weight,
            bias=None,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            activation="silu",
        ),
        # The fused CUDA MTP kernel is not the eager reference of a capture.
        gdn_decode_kernel="triton",
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_capture_static_core",
        "_forward_core_static",
        "_forward_core_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
    ):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


def _run_core(layer, fused_norm: bool, inputs: dict[str, torch.Tensor]) -> None:
    if fused_norm:
        layer._forward_core_fused_norm(**inputs)
    else:
        inputs = {name: x for name, x in inputs.items() if name != "output_gate"}
        layer._forward_core(**inputs)


def _forward_context(meta: GDNAttentionMetadata, mode: CUDAGraphMode):
    context = types.SimpleNamespace(
        attn_metadata={PREFIX: meta}, cudagraph_runtime_mode=mode
    )
    return patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or current_platform.is_device_capability_family(100),
    reason="Needs CUDA graphs and the Triton/FLA chunk kernels (not on SM10x).",
)
@pytest.mark.parametrize("fused_norm", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_captured_static_core_matches_eager(
    state_dtype: torch.dtype, fused_norm: bool
) -> None:
    from vllm.utils.torch_utils import _current_stream_tls

    torch.manual_seed(0)
    device = torch.device("cuda")
    builder = _make_builder(device)
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=NUM_SPEC
    )
    conv_state = (
        torch.randn(NUM_STATE_BLOCKS, *conv_shape, dtype=torch.bfloat16, device=device)
        * 0.05
    )
    ssm_state = (
        torch.randn(NUM_STATE_BLOCKS, *ssm_shape, dtype=state_dtype, device=device)
        * 0.05
    )
    weights = (
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
        torch.randn(V, dtype=torch.bfloat16, device=device),
    )
    captured, eager = (
        _build_layer(
            builder.vllm_config, (conv_state.clone(), ssm_state.clone()), *weights
        )
        for _ in range(2)
    )
    # Static inputs of the graph; their pad rows hold random tokens too.
    inputs = {
        "mixed_qkv": torch.empty(NUM_CAPTURED_TOKENS, CONV_DIM),
        "b": torch.empty(NUM_CAPTURED_TOKENS, HV),
        "a": torch.empty(NUM_CAPTURED_TOKENS, HV),
        "output_gate": torch.empty(NUM_CAPTURED_TOKENS, HV, V),
        "core_attn_out": torch.empty(NUM_CAPTURED_TOKENS, HV, V),
    }
    inputs = {
        name: x.to(device=device, dtype=torch.bfloat16).normal_(0, 0.1)
        for name, x in inputs.items()
    }

    prev_stream = getattr(_current_stream_tls, "value", None)
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            meta = _build(builder, "spec_and_prefill", device)
            with (
                _forward_context(meta, CUDAGraphMode.PIECEWISE),
                BreakableCUDAGraphCapture() as capture,
            ):
                # Keep the first graph segment non-empty.
                inputs["core_attn_out"].zero_()
                _run_core(captured, fused_norm, inputs)
            assert capture.num_eager_breaks == 0

            for name in BATCHES:
                meta = _build(builder, name, device)
                for tensor in inputs.values():
                    tensor.normal_(0, 0.1)
                for layer in (captured, eager):
                    layer.kv_cache[0].copy_(conv_state)
                    layer.kv_cache[1].copy_(ssm_state)
                eager_inputs = {name: x.clone() for name, x in inputs.items()}
                eager_inputs["core_attn_out"].zero_()
                capture.replay()

                with _forward_context(meta, CUDAGraphMode.NONE):
                    _run_core(eager, fused_norm, eager_inputs)
                actual = (inputs["core_attn_out"], *captured.kv_cache)
                expected = (eager_inputs["core_attn_out"], *eager.kv_cache)
                for i, (actual_tensor, expected_tensor) in enumerate(
                    zip(actual, expected)
                ):
                    if i > 0:
                        # Pad rows may write the null state block.
                        actual_tensor = actual_tensor[1:]
                        expected_tensor = expected_tensor[1:]
                    if name in NOT_BITWISE_BATCHES:
                        torch.testing.assert_close(
                            actual_tensor,
                            expected_tensor,
                            atol=NOT_BITWISE_TOLERANCE,
                            rtol=NOT_BITWISE_TOLERANCE,
                        )
                    else:
                        assert torch.equal(actual_tensor, expected_tensor), name
    finally:
        _current_stream_tls.value = prev_stream


def test_core_breaks_unless_static_form_applies() -> None:
    """The GDN core ops are captured only inside a breakable PIECEWISE capture
    whose staged capture buffers hold the batch; otherwise they break."""
    num_buffer_tokens = 16
    meta = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        capture_buffers={"non_spec_token_indx": torch.empty(num_buffer_tokens)},
    )

    def capturable(
        meta: GDNAttentionMetadata = meta,
        mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE,
        capture_active: bool = True,
        num_tokens: int = num_buffer_tokens,
        use_aiter: bool = False,
    ) -> list[bool]:
        """Decisions of the core op and the fused-norm op."""
        qkv = torch.empty(num_tokens, 1)
        with (
            _forward_context(meta, mode),
            patch.object(
                BreakableCUDAGraphCapture, "is_active", return_value=capture_active
            ),
        ):
            return [
                _qwen_gdn_attention_core_is_capturable(
                    qkv, qkv, qkv, qkv, PREFIX, use_aiter=use_aiter
                ),
                _qwen_gdn_attention_core_fused_norm_packed_is_capturable(
                    qkv, qkv, qkv, PREFIX
                ),
            ]

    assert capturable() == [True, True]
    assert capturable(capture_active=False) == [False, False]
    assert capturable(mode=CUDAGraphMode.FULL) == [False, False]
    assert capturable(num_tokens=num_buffer_tokens + 1) == [False, False]
    assert capturable(meta=replace(meta, capture_buffers=None)) == [False, False]
    # The ROCm AITER core has no static form.
    assert capturable(use_aiter=True) == [False, True]
