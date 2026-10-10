# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the host-side tensor plumbing of the GDN mixed step.

``GatedDeltaNet._forward_core`` gathers, zeroes and scatters the prefill SSM
states around ``chunk_gated_delta_rule``, and splits a batch with spec decodes
into spec and non-spec tokens. These tests run the real ``_forward_core`` on
batches built by the real ``GDNAttentionMetadataBuilder`` and require
bit-identical outputs and states against the reference op sequence.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip(
        reason="GDN _forward_core runs Triton kernels that need CUDA 8.0+.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
SPEC_TOKENS = NUM_SPEC + 1
H = 2
HV = 4
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
Z_DIM = HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"

# (seq_lens, query_lens, draft_tokens); -1 marks a request without drafts.
BATCHES = {
    "prefills": ([96, 300, 70], [96, 64, 5], [-1, -1, -1]),
    "spec-decodes-then-prefills": (
        [128, 90, 64, 200],
        [SPEC_TOKENS, 2, 64, 37],
        [NUM_SPEC, 1, -1, -1],
    ),
}
# Batches whose recurrent-update q/k/v come from rearrange_mixed_qkv.
RECURRENT_BATCHES = {
    "spec-decodes-then-prefills": BATCHES["spec-decodes-then-prefills"],
    "spec-decodes": ([128, 90], [SPEC_TOKENS, 2], [NUM_SPEC, 1]),
    "decodes-then-prefills": ([128, 90, 200], [1, 1, 37], [-1, -1, -1]),
    "decodes": ([128, 90], [1, 1], [-1, -1]),
}


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_metadata(vllm_config, batch: BatchSpec, draft_tokens: list[int]):
    device = torch.device("cuda")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    # Keep the real states off NULL_BLOCK_ID.
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        )


def _build_layer(vllm_config, kv_cache, weights):
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
        A_log=weights["A_log"],
        dt_bias=weights["dt_bias"],
        conv1d=types.SimpleNamespace(weight=weights["conv_weight"], bias=None),
        kv_cache=kv_cache,
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _make_inputs(metadata: GDNAttentionMetadata, num_tokens: int):
    """Random states, weights and a strided mixed_qkv, as split from qkvz."""
    device = torch.device("cuda")
    state_indices = [
        indices
        for indices in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if indices is not None and indices.numel() > 0
    ]
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, ssm_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state = 0.01 * torch.randn(
        pool_size, *ssm_state_shape, dtype=torch.float32, device=device
    )
    weights = {
        "A_log": 0.1 * torch.randn(HV, dtype=torch.float32, device=device),
        "dt_bias": 0.1 * torch.randn(HV, dtype=torch.float32, device=device),
        "conv_weight": 0.1
        * torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device),
    }
    mixed_qkvz = 0.1 * torch.randn(
        num_tokens, CONV_DIM + Z_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)
    return (conv_state, ssm_state), weights, mixed_qkvz, b, a


def _run_forward_core(layer, metadata, mixed_qkvz, b, a, num_tokens):
    core_attn_out = torch.zeros(
        num_tokens, HV, V, dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        layer._forward_core(
            mixed_qkv=mixed_qkvz.clone()[:, :CONV_DIM],
            b=b.clone(),
            a=a.clone(),
            core_attn_out=core_attn_out,
        )
    return core_attn_out


@pytest.mark.parametrize("batch_name", BATCHES)
@torch.inference_mode()
def test_prefill_state_gather_and_scatter_match_indexing(batch_name: str) -> None:
    """The chunk kernel sees ``ssm_state[idx]`` with fresh rows zeroed, and
    ``ssm_state[idx] = final_state`` is the only change it makes to the pool."""
    torch.manual_seed(0)
    seq_lens, query_lens, draft_tokens = BATCHES[batch_name]
    vllm_config = _make_vllm_config()
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    metadata = _build_metadata(vllm_config, batch, draft_tokens)
    assert metadata.num_prefills > 0
    prefill_state_indices = metadata.prefill_state_indices
    prefill_has_initial_state = metadata.prefill_has_initial_state
    assert prefill_state_indices is not None
    assert prefill_has_initial_state is not None
    assert not bool(prefill_has_initial_state.all())

    num_tokens = batch.compute_num_tokens()
    kv_cache, weights, mixed_qkvz, b, a = _make_inputs(metadata, num_tokens)
    ssm_state = kv_cache[1]
    # A fresh prefill must not read the stale state of its slot.
    fresh_slots = prefill_state_indices[~prefill_has_initial_state].long()
    ssm_state[fresh_slots] = float("nan")
    layer = _build_layer(vllm_config, kv_cache, weights)

    chunk_calls = []
    chunk_rule = layer.chunk_gated_delta_rule

    def recording_chunk_rule(**kwargs):
        pool_before = ssm_state.clone()
        initial_state = kwargs["initial_state"].clone()
        output, final_state = chunk_rule(**kwargs)
        chunk_calls.append((pool_before, initial_state, final_state))
        return output, final_state

    layer.chunk_gated_delta_rule = recording_chunk_rule
    _run_forward_core(layer, metadata, mixed_qkvz, b, a, num_tokens)

    assert len(chunk_calls) == 1
    pool_before, initial_state, final_state = chunk_calls[0]
    expected_initial_state = pool_before[prefill_state_indices]
    expected_initial_state[~prefill_has_initial_state, ...] = 0
    torch.testing.assert_close(
        initial_state, expected_initial_state, atol=0, rtol=0, equal_nan=True
    )
    expected_pool = pool_before.clone()
    expected_pool[prefill_state_indices] = final_state.to(expected_pool.dtype)
    torch.testing.assert_close(ssm_state, expected_pool, atol=0, rtol=0)


@torch.inference_mode()
def test_spec_token_slices_match_gathers() -> None:
    """Spec-first batches slice the spec and non-spec tokens; the result is
    bit-identical to the index_select / index_copy_ path."""
    torch.manual_seed(0)
    seq_lens, query_lens, draft_tokens = BATCHES["spec-decodes-then-prefills"]
    vllm_config = _make_vllm_config()
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    metadata = _build_metadata(vllm_config, batch, draft_tokens)
    assert metadata.spec_tokens_first
    gather_metadata = dataclasses.replace(metadata, spec_tokens_first=False)

    num_tokens = batch.compute_num_tokens()
    kv_cache, weights, mixed_qkvz, b, a = _make_inputs(metadata, num_tokens)
    results = []
    for step_metadata in (metadata, gather_metadata):
        step_kv_cache = tuple(state.clone() for state in kv_cache)
        layer = _build_layer(vllm_config, step_kv_cache, weights)
        out = _run_forward_core(layer, step_metadata, mixed_qkvz, b, a, num_tokens)
        results.append((out, *step_kv_cache))

    for sliced, gathered in zip(*results):
        torch.testing.assert_close(sliced, gathered, atol=0, rtol=0)


def _rearrange_mixed_qkv_copies(layer, mixed_qkv, *, contiguous):
    return QwenGatedDeltaNetAttention.rearrange_mixed_qkv(layer, mixed_qkv)


@pytest.mark.parametrize("batch_name", RECURRENT_BATCHES)
@torch.inference_mode()
def test_strided_qkv_views_match_contiguous_copies(batch_name: str) -> None:
    """The recurrent update gives bit-identical results for the strided q/k/v
    views and for the contiguous q/k/v copies of rearrange_mixed_qkv."""
    torch.manual_seed(0)
    seq_lens, query_lens, draft_tokens = RECURRENT_BATCHES[batch_name]
    vllm_config = _make_vllm_config()
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    metadata = _build_metadata(vllm_config, batch, draft_tokens)

    num_tokens = batch.compute_num_tokens()
    kv_cache, weights, mixed_qkvz, b, a = _make_inputs(metadata, num_tokens)
    results = []
    for contiguous in (False, True):
        step_kv_cache = tuple(state.clone() for state in kv_cache)
        layer = _build_layer(vllm_config, step_kv_cache, weights)
        if contiguous:
            # Ignore the call site's contiguous=False: copy as before.
            layer.rearrange_mixed_qkv = functools.partial(
                _rearrange_mixed_qkv_copies, layer
            )
        out = _run_forward_core(layer, metadata, mixed_qkvz, b, a, num_tokens)
        results.append((out, *step_kv_cache))

    for strided, copied in zip(*results):
        torch.testing.assert_close(strided, copied, atol=0, rtol=0)
