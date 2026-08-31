# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for causal masking in the AITER MLA verify MTP path.

When ``num_heads < 16`` the ROCm AITER MLA backend cannot use the ASM decode
kernel for a multi-token block, so ``AiterMLAImpl.forward_mqa`` routes verify
through ``mla_gluon``'s native 4-D MTP entry (``q`` shaped
``[batch, qlen, nhead, dim]`` with ``use_2d_view=False``). aiter applies the
causal tail internally: position ``t`` attends KV ``[0, seq_len - qlen + t]``.

The test drives the real builder and ``forward_mqa`` over a multi-token decode
batch, intercepts the arguments actually handed to the Gluon kernel, and pins
that the 4-D MTP contract is used with per-request ``block_table`` and
``cache_seqlens`` rather than a flattened per-token paged-KV view.
"""

import types
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform


def _on_rocm_with_aiter() -> bool:
    return current_platform.is_rocm() and is_aiter_found()


# The production verify MTP path is gfx950-only. This test replaces the Gluon
# kernel with a spy and forces the architecture feature probe so the metadata
# regression remains testable on every ROCm AITER CI runner.
pytestmark = pytest.mark.skipif(
    not _on_rocm_with_aiter(),
    reason="ROCM_AITER_MLA verify MTP requires ROCm and AITER",
)

# Below the 16-head threshold that selects the MTP path. DeepSeek-V3 / R1 at
# TP=16 and Kimi-K3 at TP=8 both land here.
NUM_QUERY_HEADS = 12
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM

# 1 + num_speculative_tokens, i.e. the verify block length.
QLEN = 8
# Committed context per request, including a fresh request at zero context
# where the whole KV range is the verify block itself.
CONTEXT_LENS = [0, 1, 37, 512]
# A cudagraph padding request: seq_len 0.
PADDING_ROWS = 1

# The Gluon path flattens the KV cache to one page per token.
PAGE_SIZE = 1
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 8

# The builder resolves the verify routing and records it on the metadata, so
# the gfx950 feature probe has to be forced for the build as well as for the
# forward; ROCm CI also runs gfx942, where the real probe is False.
_GLUON_SUPPORTED_TARGET = (
    "vllm.v1.attention.backends.mla.rocm_aiter_mla._gluon_mla_decode_supported"
)


def _seq_lens() -> list[int]:
    return [c + QLEN for c in CONTEXT_LENS] + [0] * PADDING_ROWS


def _run_verify_block():
    """Drive the real builder + forward_mqa over one multi-token verify block.

    Returns ``(metadata, captured)`` where ``captured`` holds the Gluon kwargs.
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
    vllm_config.model_config.get_num_attention_heads = types.MethodType(
        lambda self, parallel_config, arch_config=None: NUM_QUERY_HEADS,
        vllm_config.model_config,
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
        types.SimpleNamespace(
            prefill_backend=torch.empty((1,)),
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )
    )

    init_workspace_manager(device)

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=[QLEN] * num_reqs)

    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)
        captured["q_nope"] = kwargs["q_nope"].detach().clone()
        captured["page_table"] = kwargs["page_table"].detach().clone()
        captured["seq_info"] = kwargs["seq_info"].detach().clone()

    with (
        patch(_GLUON_SUPPORTED_TARGET, lambda: True),
        set_current_vllm_config(vllm_config),
    ):
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

    # The Gluon kernel only reads the metadata this test is about, so a spy in
    # its place keeps the assertions independent of the AITER build.
    with (
        patch(_GLUON_SUPPORTED_TARGET, lambda: True),
        patch(
            "vllm.v1.attention.backends.mla.rocm_aiter_mla._get_mla_gluon",
            lambda: spy,
        ),
    ):
        impl.forward_mqa((q_nope, q_pe), kv_cache, metadata, layer=None)

    return metadata, captured


def test_verify_mtp_uses_native_4d_gluon_entry():
    """Verify uses 4-D q with per-request paged_kv metadata.

    Regression guard: before the fix every flattened row of a request got that
    request's whole paged-KV range, so verify position t attended to the draft
    tokens after it. The native MTP path delegates causal masking to aiter.
    """
    metadata, captured = _run_verify_block()

    decode = metadata.decode
    assert decode is not None, "batch was not classified as a decode"
    assert decode.max_qo_len == QLEN, (
        f"expected a {QLEN}-token verify block, got max_qo_len="
        f"{decode.max_qo_len}; the MTP path under test was not reached"
    )
    assert decode.use_gluon_verify, (
        "the builder owns verify routing, so the metadata -- not the impl -- "
        "has to select the Gluon MTP entry for a small-head bf16 verify block"
    )
    assert captured, "forward_mqa did not reach the Gluon kernel"

    seq_lens = _seq_lens()
    num_reqs = len(seq_lens)

    q_nope = captured["q_nope"]
    assert q_nope.dim() == 4, (
        f"verify must pass 4-D q to mla_gluon, got {q_nope.dim()}-D shape "
        f"{tuple(q_nope.shape)}"
    )
    assert q_nope.shape[:2] == (num_reqs, QLEN), (
        f"expected q shape [num_reqs={num_reqs}, qlen={QLEN}, ...], "
        f"got {tuple(q_nope.shape)}"
    )
    assert q_nope.shape[2] == NUM_QUERY_HEADS

    assert captured["use_2d_view"] is False
    page_table = captured["page_table"]
    seq_info = captured["seq_info"]
    assert page_table.dim() == 1, (
        f"verify MTP passes the 1-D paged_kv_indices buffer, got {page_table.dim()}-D"
    )
    assert seq_info.dim() == 1 and seq_info.numel() == num_reqs + 1, (
        f"verify MTP passes paged_kv_indptr [num_reqs + 1], got shape "
        f"{tuple(seq_info.shape)}"
    )
    assert decode is not None and decode.paged_kv_indptr is not None
    assert torch.equal(seq_info, decode.paged_kv_indptr[: num_reqs + 1])
    assert torch.equal(
        page_table[: int(seq_info[-1].item())],
        decode.paged_kv_indices[: int(seq_info[-1].item())],
    )
