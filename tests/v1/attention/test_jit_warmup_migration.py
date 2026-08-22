# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate JIT dispatch against pre-contract behavior."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.v1.attention.backends.mla.sparse_swa import ComputePrefillMetadataKernel
from vllm.v1.attention.ops.dcp import CorrectAttnCPOutKernel

_HAS_CUTEDSL = has_cutedsl()
requires_cutedsl = pytest.mark.skipif(
    not _HAS_CUTEDSL,
    reason="CuTeDSL is not installed",
)

if _HAS_CUTEDSL:
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        StableTopKFromGatheredCandidatesKernel,
    )


def test_correct_attn_cp_out_warmup_uses_dsv4_head_dim() -> None:
    hf_config = SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        num_attention_heads=128,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=hf_config,
            hf_text_config=hf_config,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            tensor_parallel_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
    )

    warmup_keys = CorrectAttnCPOutKernel().get_warmup_keys(vllm_config)

    assert len(warmup_keys) == 4
    assert {key.head_dim for key in warmup_keys} == {512}


@pytest.mark.parametrize(
    ("num_prefills", "expected_block_size"),
    [(1, 1), (3, 4), (8, 8), (9, 16)],
)
def test_compute_prefill_metadata_dispatch_matches_legacy_meta(
    num_prefills: int,
    expected_block_size: int,
) -> None:
    kernel = ComputePrefillMetadataKernel()

    assert kernel.dispatch(num_prefills=num_prefills) == kernel.CompileKey(
        block_size=expected_block_size
    )


@requires_cutedsl
def test_stable_topk_dispatch_matches_legacy_compile_args() -> None:
    kernel = StableTopKFromGatheredCandidatesKernel()

    assert kernel.dispatch(topk=512, num_candidates=2048) == kernel.CompileKey(
        topk=512,
        num_candidates=2048,
    )
