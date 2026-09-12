# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSv4 indexer must not hand a compress-128 page to DeepGEMM.

``get_paged_mqa_logits_metadata`` asserts ``block_kv in {32, 64}`` host-side.
``num_states`` is ``block_size // tokens_per_state``, and the indexer kernel
block size is fixed at 256, so a C4A page (``tokens_per_state=4``) resolves to
the supported 64 states while a compress-128 page resolves to 2 and trips the
assert inside ``DeepseekV32IndexerMetadataBuilder.build()``.

``_should_build_paged_mqa_logits_metadata`` is covered directly elsewhere; these
tests pin the behaviour at the call site, which is the only place the bad value
can reach the kernel.
"""

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# The only kernel block size the DSv4 indexer advertises.
BLOCK_SIZE = 256
# num_states = BLOCK_SIZE // tokens_per_state
C4A_TOKENS_PER_STATE = 4  # -> 64 states, accepted by DeepGEMM
C128_TOKENS_PER_STATE = 128  # -> 2 states, rejected by DeepGEMM


def _make_spec(tokens_per_state: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=tokens_per_state,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
        state_content_bytes=584,
    )


def test_kernel_block_size_and_state_arithmetic():
    """The premise of the gate, with no device required."""
    assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [BLOCK_SIZE]
    assert _make_spec(C4A_TOKENS_PER_STATE).num_states == 64
    assert _make_spec(C128_TOKENS_PER_STATE).num_states == 2


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the indexer decode path is CUDA only"
)
@pytest.mark.parametrize(
    "tokens_per_state,expect_deep_gemm_call",
    [(C4A_TOKENS_PER_STATE, True), (C128_TOKENS_PER_STATE, False)],
)
def test_decode_build_skips_deep_gemm_for_two_state_pages(
    monkeypatch, tokens_per_state, expect_deep_gemm_call
):
    """A 2-state page must not reach ``get_paged_mqa_logits_metadata``.

    The DeepGEMM entry point is stubbed, so this asserts which branch ran rather
    than relying on the kernel to raise. Without the gate the compress-128 case
    calls through and the recorded value is 2.
    """
    device = torch.device("cuda")
    spec = _make_spec(tokens_per_state)

    calls: list[int] = []

    def _record(seq_lens, num_states, num_sms, indices=None):
        calls.append(num_states)
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=seq_lens.device)

    monkeypatch.setattr(indexer, "get_paged_mqa_logits_metadata", _record)
    # raising=False so the failure below is the routing decision rather than a
    # missing symbol when the module has not imported the support helper.
    monkeypatch.setattr(indexer, "is_deep_gemm_supported", lambda: True, raising=False)

    builder = _build_builder(spec, device)
    builder.build(0, _decode_only_metadata(device))

    assert calls == ([spec.num_states] if expect_deep_gemm_call else [])


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepGEMM paged MQA metadata is CUDA only"
)
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM is not available")
@pytest.mark.parametrize(
    "tokens_per_state", [C4A_TOKENS_PER_STATE, C128_TOKENS_PER_STATE]
)
def test_decode_build_survives_compress_128_on_device(tokens_per_state):
    """End to end against the real kernel entry point.

    Without the gate the compress-128 case raises
    ``block_kv == 32 or block_kv == 64`` from ``attention.hpp``.
    """
    device = torch.device("cuda")
    builder = _build_builder(_make_spec(tokens_per_state), device)

    metadata = builder.build(0, _decode_only_metadata(device))

    assert metadata.decode is not None
    assert metadata.decode.schedule_metadata.shape[1] == 2


def _build_builder(spec: MLAAttentionSpec, device: torch.device):
    from tests.v1.attention.utils import create_vllm_config

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=BLOCK_SIZE * 4,
        block_size=BLOCK_SIZE,
        max_num_seqs=4,
        max_num_batched_tokens=BLOCK_SIZE,
    )
    builder_cls = DeepseekV4IndexerBackend.get_builder_cls()
    return builder_cls(
        kv_cache_spec=spec,
        layer_names=["placeholder"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=BLOCK_SIZE // 4,
    )


def _decode_only_metadata(device: torch.device):
    batch_spec = BatchSpec(seq_lens=[BLOCK_SIZE, BLOCK_SIZE], query_lens=[1, 1])
    return create_common_attn_metadata(batch_spec, BLOCK_SIZE, device)
