# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for native causal AITER MLA verification.

Gluon accepts a uniform multi-token query as ``[batch, qlen, heads, dim]`` and
applies the causal tail internally. Keep the query dimension intact and pass
the builder's per-request KV metadata through unchanged; flattening the query
to independent decode rows loses those causal semantics.
"""

import types
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform


def _on_rocm_with_aiter() -> bool:
    return current_platform.is_rocm() and is_aiter_found()


# This path is selected by head count rather than architecture. The Gluon
# kernel is replaced by a spy, so the test only requires AITER to construct the
# real backend and metadata.
pytestmark = pytest.mark.skipif(
    not _on_rocm_with_aiter(),
    reason="ROCM_AITER_MLA native verification requires ROCm and AITER",
)

# Below the 16-head threshold that selects Gluon. Kimi-K3 TP8 lands here.
NUM_QUERY_HEADS = 12
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM

# 1 + num_speculative_tokens, i.e. the verify block length.
QLEN = 8
# Committed context per request, including a fresh request at zero context.
CONTEXT_LENS = [0, 1, 37, 512]
# Include one cudagraph padding request.
PADDING_ROWS = 1

# Gluon uses one cache page per token.
PAGE_SIZE = 1
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 8


def _seq_lens() -> list[int]:
    return [c + QLEN for c in CONTEXT_LENS] + [0] * PADDING_ROWS


def _run_verify_block():
    """Drive the real builder + forward_mqa over one multi-token verify block.

    Returns the metadata and the arguments passed to the Gluon kernel.
    """
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import SpeculativeConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda:0")
    seq_lens = _seq_lens()
    num_reqs = len(seq_lens)
    # One flat page per token, arange block indices, plus room to spare.
    num_gpu_blocks = num_reqs * max(seq_lens) + 256

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        max_model_len=MAX_MODEL_LEN,
        num_gpu_blocks=num_gpu_blocks,
        block_size=PAGE_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=8192,
        hf_config_override={"num_attention_heads": NUM_QUERY_HEADS},
    )
    # What raises reorder_batch_threshold to QLEN, so a QLEN-token block is
    # classified as a decode instead of a prefill. ngram needs no draft model.
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=QLEN - 1
    )

    spec = MLAAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=1,
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
        cache_dtype_str="auto",
    )

    backend_cls = AttentionBackendEnum.ROCM_AITER_MLA.get_class()
    builder_cls = backend_cls.get_builder_cls()
    impl_cls = backend_cls.get_impl_cls()

    # The builder reads layer.prefill_backend from static_forward_context; a
    # stub with the attribute is enough for metadata construction.
    layer_name = "placeholder"
    vllm_config.compilation_config.static_forward_context[layer_name] = (
        types.SimpleNamespace(prefill_backend=torch.empty((1,)))
    )

    init_workspace_manager(device)

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=[QLEN] * num_reqs)

    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)

    with set_current_vllm_config(vllm_config):
        builder = builder_cls(spec, [layer_name], vllm_config, device)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, PAGE_SIZE, device, arange_block_indices=True
        )
        metadata = builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        impl = impl_cls(
            num_heads=NUM_QUERY_HEADS,
            head_size=HEAD_SIZE,
            scale=(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=None,
        )

    num_rows = num_reqs * QLEN
    dtype = torch.bfloat16
    q_nope = torch.zeros(
        num_rows, NUM_QUERY_HEADS, KV_LORA_RANK, dtype=dtype, device=device
    )
    q_pe = torch.zeros(
        num_rows, NUM_QUERY_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        num_gpu_blocks, PAGE_SIZE, HEAD_SIZE, dtype=dtype, device=device
    )

    # A spy keeps the dispatch assertions independent of Gluon numerics.
    with patch(
        "vllm.v1.attention.backends.mla.rocm_aiter_mla._get_mla_gluon",
        lambda: spy,
    ):
        impl.forward_mqa((q_nope, q_pe), kv_cache, metadata, layer=None)

    return metadata, captured


def test_verify_uses_native_causal_gluon():
    metadata, captured = _run_verify_block()

    decode = metadata.decode
    assert decode is not None, "batch was not classified as a decode"
    assert decode.max_qo_len == QLEN, (
        f"expected a {QLEN}-token verify block, got max_qo_len="
        f"{decode.max_qo_len}; native verification was not reached"
    )
    assert captured, "forward_mqa did not reach the Gluon kernel"
    num_reqs = len(_seq_lens())
    assert captured["q_nope"].shape == (
        num_reqs,
        QLEN,
        NUM_QUERY_HEADS,
        KV_LORA_RANK,
    )
    assert captured["q_pe"].shape == (
        num_reqs,
        QLEN,
        NUM_QUERY_HEADS,
        QK_ROPE_HEAD_DIM,
    )
    assert captured["o"].shape == (
        num_reqs,
        QLEN,
        NUM_QUERY_HEADS,
        KV_LORA_RANK,
    )
    assert captured["page_table"] is decode.paged_kv_indices
    assert captured["seq_info"] is decode.paged_kv_indptr
    assert captured["min_kv_seq_len"] == decode.min_kv_seq_len
