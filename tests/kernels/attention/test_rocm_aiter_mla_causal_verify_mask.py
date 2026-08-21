# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for causal masking in the AITER MLA verify flatten.

When ``num_heads < 16`` the ROCm AITER MLA backend cannot use the ASM decode
kernel for a multi-token block, so ``AiterMLAImpl.forward_mqa`` flattens a
``1 + num_speculative_tokens`` verify block into that many single-token Gluon
decodes. Every resulting row used to be handed its request's entire paged-KV
range. ``paged_kv_indptr`` is the cumulative sum of ``seq_lens``, which counts
the tokens scheduled in the current step, so that range already spans the
verify block: each verify position could attend to the draft tokens after it,
which are the tokens it is supposed to be checking. Nothing crashes and nothing
warns, so the only way to catch it is to look at the KV window per row.

The test drives the real builder and the real ``forward_mqa`` over a
multi-token decode batch, intercepts the ``(page_table, seq_info)`` actually
handed to the Gluon kernel, and pins verify row ``t`` of request ``r`` to the
``context_r + t + 1`` entries of that request's ascending slice.
"""

import types
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform


def _on_rocm_with_aiter() -> bool:
    return current_platform.is_rocm() and is_aiter_found()


# The production verify flatten is gfx950-only. This test replaces the Gluon
# kernel with a spy and forces the architecture feature probe so the metadata
# regression remains testable on every ROCm AITER CI runner.
pytestmark = pytest.mark.skipif(
    not _on_rocm_with_aiter(),
    reason="ROCM_AITER_MLA verify flatten requires ROCm and AITER",
)

# Below the 16-head threshold that selects the flatten. DeepSeek-V3 / R1 at
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
# A cudagraph padding request: seq_len 0, so every one of its rows must clamp
# to an empty window.
PADDING_ROWS = 1

# Distinguishable from 1.0 so the fp8 route's scale handling is observable.
Q_SCALE = 0.5
K_SCALE = 0.25

# The Gluon path flattens the KV cache to one page per token.
PAGE_SIZE = 1
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 8


def _seq_lens() -> list[int]:
    return [c + QLEN for c in CONTEXT_LENS] + [0] * PADDING_ROWS


def _expected_row_lens() -> list[int]:
    """Causal row lengths: row r*QLEN + t sees seq_len_r - (QLEN - 1) + t."""
    return [max(0, s - (QLEN - 1) + t) for s in _seq_lens() for t in range(QLEN)]


def _run_verify_block(
    cache_dtype_str: str = "auto",
    run_forward: bool = True,
    full_cudagraphs: bool = True,
):
    """Drive the real builder + forward_mqa over one multi-token verify block.

    ``cache_dtype_str`` selects the KV cache the flatten runs over: ``"auto"``
    hands Gluon a bf16 query and cache, ``"fp8"`` an fp8 cache and -- as the
    layer would -- an fp8 query the flatten has to dequantize itself.

    Returns ``(metadata, captured)`` where ``captured`` holds the ``page_table``,
    ``seq_info``, ``min_kv_seq_len``, the query dtype and the cache scale the
    backend passed to the Gluon kernel.
    """
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import CUDAGraphMode, SpeculativeConfig
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
    # get_num_attention_heads() reads model_arch_config, which is derived when
    # ModelConfig is built, so the hf_config override alone would leave the
    # builder sizing the decode for the checkpoint's 128 heads while the impl
    # below runs with NUM_QUERY_HEADS.
    vllm_config.model_config.model_arch_config.total_num_attention_heads = (
        NUM_QUERY_HEADS
    )
    if not full_cudagraphs:
        # Gluon's split count is fixed at launch, so the builder only passes the
        # real per-step minimum row length when nothing captures it.
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    kv_dtype = (
        torch.float8_e4m3fn
        if cache_dtype_str == "fp8"
        else vllm_config.model_config.dtype
    )
    vllm_config.cache_config.cache_dtype = cache_dtype_str
    spec = MLAAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=1,
        head_size=vllm_config.model_config.get_head_size(),
        dtype=kv_dtype,
        cache_dtype_str=cache_dtype_str,
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
        captured["page_table"] = kwargs["page_table"].detach().clone()
        captured["seq_info"] = kwargs["seq_info"].detach().clone()
        captured["min_kv_seq_len"] = kwargs["min_kv_seq_len"]
        captured["q_dtype"] = kwargs["q_nope"].dtype
        captured["kv_scale"] = kwargs["kv_scale"]

    # The arch and aiter-feature probes are forced for the builder as well as for
    # the forward below: the builder takes the per-row KV ranges, so it has to
    # reach the same routing decision the impl acts on.
    def _gluon_probes() -> list:
        prefix = "vllm.v1.attention.backends.mla.rocm_aiter_mla."
        return [
            patch(prefix + "_gluon_mla_decode_supported", lambda: True),
            patch(prefix + "_gluon_mla_fp8_batch_supported", lambda: True),
        ]

    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(vllm_config))
        for probe in _gluon_probes():
            stack.enter_context(probe)
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
            kv_cache_dtype=cache_dtype_str,
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

    if not run_forward:
        # The routing decision is already visible in the metadata: the builder
        # only fills the per-row view when the flatten will serve the batch.
        return metadata, captured

    num_rows = num_reqs * QLEN
    # An fp8 KV cache is handed an fp8 query, as MLAAttention would; the flatten
    # is what widens it back for Gluon.
    q_dtype = torch.float8_e4m3fn if cache_dtype_str == "fp8" else torch.bfloat16
    q_nope = torch.zeros(
        num_rows, NUM_QUERY_HEADS, KV_LORA_RANK, dtype=q_dtype, device=device
    )
    q_pe = torch.zeros(
        num_rows, NUM_QUERY_HEADS, QK_ROPE_HEAD_DIM, dtype=q_dtype, device=device
    )
    kv_cache = torch.zeros(
        num_gpu_blocks, PAGE_SIZE, HEAD_SIZE, dtype=kv_dtype, device=device
    )
    layer = types.SimpleNamespace(
        _q_scale=torch.tensor([Q_SCALE], dtype=torch.float32, device=device),
        _k_scale_float=K_SCALE,
    )

    # The Gluon kernel only reads the metadata this test is about, so a spy in
    # its place keeps the assertions independent of the AITER build.
    with ExitStack() as stack:
        for probe in _gluon_probes():
            stack.enter_context(probe)
        stack.enter_context(
            patch(
                "vllm.v1.attention.backends.mla.rocm_aiter_mla._get_mla_gluon",
                lambda: spy,
            )
        )
        impl.forward_mqa((q_nope, q_pe), kv_cache, metadata, layer=layer)

    return metadata, captured


def test_verify_flatten_rows_are_causal():
    """Verify row t must see the committed prefix plus tokens 0..t, and no more.

    Regression guard: before the fix every row of a request got that request's
    whole paged-KV range, so verify position t attended to the draft tokens
    after it. Fails on unmodified upstream.
    """
    metadata, captured = _run_verify_block()

    decode = metadata.decode
    assert decode is not None, "batch was not classified as a decode"
    assert decode.max_qo_len == QLEN, (
        f"expected a {QLEN}-token verify block, got max_qo_len="
        f"{decode.max_qo_len}; the flatten under test was not reached"
    )
    assert captured, "forward_mqa did not reach the Gluon kernel"

    seq_lens = _seq_lens()
    indptr = captured["seq_info"].tolist()
    page_table = captured["page_table"]
    got_row_lens = [indptr[i + 1] - indptr[i] for i in range(len(indptr) - 1)]
    want_row_lens = _expected_row_lens()

    assert got_row_lens == want_row_lens, (
        "AITER MLA verify flatten is not causal: verify row r*qlen+t must get "
        f"seq_len_r - {QLEN - 1} + t KV entries.\n"
        f"  seq_lens   {seq_lens}\n"
        f"  got        {got_row_lens}\n"
        f"  expected   {want_row_lens}\n"
        "Rows longer than expected let a verify position attend to the draft "
        "tokens it is supposed to be checking."
    )

    # arange_block_indices lays request r's pages out ascending from
    # r * max_blocks, so each causal window must be that slice's prefix.
    max_blocks = max(seq_lens)
    for r, seq_len in enumerate(seq_lens):
        for t in range(QLEN):
            row = r * QLEN + t
            window = page_table[indptr[row] : indptr[row + 1]].tolist()
            n = max(0, seq_len - (QLEN - 1) + t)
            want = [r * max_blocks + j for j in range(n)]
            assert window == want, (
                f"request {r} (seq_len {seq_len}) verify token {t} reads KV "
                f"pages {window}, expected the ascending causal prefix {want}"
            )

    padding_rows = got_row_lens[len(CONTEXT_LENS) * QLEN :]
    assert padding_rows == [0] * (PADDING_ROWS * QLEN), (
        f"cudagraph padding requests (seq_len 0) must clamp to empty windows, "
        f"got {padding_rows}"
    )

    # Gluon sizes its KV split count from this bound, and a captured graph fixes
    # that count, so a replay whose shortest row is shorter would read an empty
    # split. Under full cudagraphs the builder pins it to a single split.
    assert captured["min_kv_seq_len"] == 1, (
        f"min_kv_seq_len={captured['min_kv_seq_len']} should be pinned to a "
        "single split while a graph can capture it"
    )


def test_verify_flatten_min_row_len_is_exact_when_eager():
    """Without a graph to freeze it, the bound is the shortest submitted row.

    It must be the minimum over the causal rows, not over the per-request
    lengths they were cut from, or Gluon splits a row it cannot fill.
    """
    _, captured = _run_verify_block(full_cudagraphs=False)
    want = min(_expected_row_lens())
    assert captured["min_kv_seq_len"] == want, (
        f"min_kv_seq_len={captured['min_kv_seq_len']} does not match the "
        f"shortest submitted row ({want})"
    )


def test_verify_flatten_keeps_a_bf16_query_unscaled():
    """A bf16 cache hands Gluon the query as-is, with no scale folded in."""
    _, captured = _run_verify_block(cache_dtype_str="auto")
    assert captured["q_dtype"] == torch.bfloat16
    assert captured["kv_scale"] == 1.0, (
        f"a bf16 KV cache needs no dequant scale, got {captured['kv_scale']}"
    )


def test_fp8_verify_stays_on_the_asm_decode_without_dcp():
    """Without DCP an fp8 verify keeps the asm decode, which has a kernel for it.

    The flatten serves fp8 only where there is no alternative, i.e. under DCP.
    Widening it further would move every fp8 small-head verify onto a Gluon path
    that has not been measured against the asm q-row fold.
    """
    metadata, _ = _run_verify_block(cache_dtype_str="fp8", run_forward=False)
    decode = metadata.decode
    assert decode is not None, "batch was not classified as a decode"
    assert decode.verify_row_lens is None, (
        "the builder took the Gluon flatten for an fp8 verify without DCP; the "
        "asm verify serves that batch"
    )


@pytest.mark.parametrize("fp8_batch_supported", [True, False])
def test_fp8_verify_flatten_gate(fp8_batch_supported: bool):
    """Under DCP the fp8 flatten follows Gluon's own batch support.

    A single process cannot hold a DCP group, so the routing decision is checked
    on the predicate the builder and the impl both call.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    prefix = "vllm.v1.attention.backends.mla.rocm_aiter_mla."
    with (
        patch(prefix + "_gluon_mla_decode_supported", lambda: True),
        patch(prefix + "_gluon_mla_max_bh16_heads", lambda: 96),
        patch(prefix + "_gluon_mla_fp8_batch_supported", lambda: fp8_batch_supported),
    ):
        # TP8 x DCP8 gathers 96 heads onto the decode kernel.
        got = AiterMLAHelper.use_gluon_verify(96, QLEN, "fp8", 8)
    assert got is fp8_batch_supported, (
        "an fp8 DCP verify must take the flatten exactly when Gluon's fp8 regime "
        f"accepts a batch (support={fp8_batch_supported}, took flatten={got})"
    )
