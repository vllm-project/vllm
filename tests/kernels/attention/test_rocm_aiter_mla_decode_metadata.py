# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for AITER MLA persistent decode metadata dtypes.

For the gfx950 fp8/fp8 nhead=32 qlen=1 fold path, the split/reduce metadata
layout depends on the q/kv element size. The builder must forward dtype_q/dtype_kv
to ``get_mla_metadata_v1``; omitting them lays out the work for the wrong dtype
and corrupts decode output. The test pins the builder's metadata to a golden
recomputed at runtime with the explicit correct dtypes.
"""

import types
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not (current_platform.is_rocm() and is_aiter_found()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(),
    reason="AITER MLA fp8 persistent decode metadata is gfx950-only",
)

# The fold path that the bug corrupted: fp8 query + fp8 KV-cache, 32 query
# heads, single-token decode, batch 128, context 8192, page_size 1.
NUM_QUERY_HEADS = 32
DECODE_QLEN = 1
BATCH_SIZE = 128
CONTEXT_LEN = 8192
PAGE_SIZE = 1

# Expected dtypes for this fold path: bf16 model dtype -> bf16 query; fp8
# KV-cache -> fp8_e4m3 kv.
EXPECTED_Q_DTYPE = torch.bfloat16
EXPECTED_KV_DTYPE = torch.float8_e4m3fn

# The split/reduce content tensors filled by get_mla_metadata_v1. work_meta_data
# is excluded: it holds raw device pointers, never equal across allocations.
_CONTENT_METADATA_FIELDS = (
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)

# The builder's get_mla_metadata_v1 call passes 6 input args then 6 output
# buffers (see AiterMLAMetadataBuilder._build_decode). Output order -> field.
_NUM_INPUT_ARGS = 6
_OUTPUT_ARG_FIELDS = (
    "work_meta_data",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _build_decode_metadata():
    """Build AITER MLA decode metadata for the fp8/fp8 nhead=32 fold path.

    Returns ``(metadata, captured)`` where ``captured`` records the positional
    args/kwargs the builder passed to ``get_mla_metadata_v1``, so the golden can
    be recomputed from the identical inputs.
    """
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config.vllm import set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda:0")

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        max_model_len=CONTEXT_LEN,
        # One flat page per token (page_size=1); +buffer for the null block.
        num_gpu_blocks=BATCH_SIZE * CONTEXT_LEN + 200,
        block_size=PAGE_SIZE,
        max_num_seqs=BATCH_SIZE,
        max_num_batched_tokens=8192,
        hf_config_override={"num_attention_heads": NUM_QUERY_HEADS},
    )
    vllm_config.cache_config.cache_dtype = "fp8"

    spec = MLAAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=1,
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
        cache_dtype_str="fp8",
    )

    builder_cls = AttentionBackendEnum.ROCM_AITER_MLA.get_class().get_builder_cls()

    # The builder reads layer.prefill_backend from static_forward_context; a
    # stub with the attribute is enough for metadata construction.
    layer_name = "placeholder"
    vllm_config.compilation_config.static_forward_context[layer_name] = (
        types.SimpleNamespace(prefill_backend=torch.empty((1,)))
    )

    init_workspace_manager(device)

    batch_spec = BatchSpec(
        seq_lens=[CONTEXT_LEN] * BATCH_SIZE,
        query_lens=[DECODE_QLEN] * BATCH_SIZE,
    )

    captured: dict = {}

    with set_current_vllm_config(vllm_config):
        builder = builder_cls(spec, [layer_name], vllm_config, device)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, PAGE_SIZE, device, arange_block_indices=True
        )

        import aiter

        real_get_mla_metadata_v1 = aiter.get_mla_metadata_v1

        def spy(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            return real_get_mla_metadata_v1(*args, **kwargs)

        with patch("aiter.get_mla_metadata_v1", spy):
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

    return metadata, captured


def _compute_golden_metadata(captured: dict) -> dict[str, torch.Tensor]:
    """Recompute the persistent metadata with explicit fp8/bf16 dtypes.

    Replays ``get_mla_metadata_v1`` on the builder's exact input tensors with
    fresh output buffers and the explicitly-correct dtypes. This reference must
    match the builder's output when the fix is in place.
    """
    import aiter

    args = captured["args"]
    inputs = args[:_NUM_INPUT_ARGS]
    # Fresh copies so the golden does not alias the builder's persistent buffers.
    fresh_outputs = [arg.clone() for arg in args[_NUM_INPUT_ARGS:]]

    golden_kwargs = dict(captured["kwargs"])
    golden_kwargs["dtype_q"] = EXPECTED_Q_DTYPE
    golden_kwargs["dtype_kv"] = EXPECTED_KV_DTYPE

    aiter.get_mla_metadata_v1(*inputs, *fresh_outputs, **golden_kwargs)

    return dict(zip(_OUTPUT_ARG_FIELDS, fresh_outputs))


def test_persistent_decode_metadata_matches_fp8_golden():
    """The builder's metadata must match the dtype-correct golden.

    Regression guard: the fixed builder forwards fp8/bf16 dtypes so its
    split/reduce metadata matches the golden recomputed with those explicit
    dtypes. Dropping the dtypes (the original bug) produces a different layout
    and fails this test.
    """
    metadata, captured = _build_decode_metadata()

    # qlen=1 must take the persistent-metadata path for this to be meaningful.
    assert metadata.decode is not None
    assert metadata.decode.has_persistent_metadata
    assert metadata.work_meta_data is not None

    golden = _compute_golden_metadata(captured)

    mismatched = [
        name
        for name in _CONTENT_METADATA_FIELDS
        if getattr(metadata, name).shape != golden[name].shape
        or not torch.equal(getattr(metadata, name), golden[name])
    ]
    assert not mismatched, (
        "AITER MLA persistent decode metadata does not match the fp8/bf16 "
        f"golden for fields {mismatched}; the builder must forward "
        "dtype_q/dtype_kv to get_mla_metadata_v1."
    )
