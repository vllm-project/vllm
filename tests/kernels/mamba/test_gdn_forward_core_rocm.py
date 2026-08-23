# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``QwenGatedDeltaNetAttention._forward_core_rocm``.

``forward_hip`` allocates ``core_attn_out`` with ``torch.empty``, so the
wrapper owes every row that is read later a definite value before it hands the
buffer to ``_forward_core``. On a pure non-spec decode batch ``_forward_core``
reaches the packed decode kernel, which fills ``core_attn_out``'s first
``num_actual_tokens`` rows itself, so the fill is redundant once the batch
covers the whole buffer. Every other batch shape can leave rows untouched and
keeps the full fill.

The tests pin the buffer contents the wrapper produces rather than how it
produces them, so they survive a restructuring of the fill, and they guard the
fast-path predicate against drifting away from the dispatch it mirrors. All but
``test_packed_decode_kernel_writes_every_row_it_owns`` run on CPU, with
everything below the wrapper stubbed out.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

PREFIX = "model.layers.0.linear_attn"
HV = 2  # num value heads
V = 4  # head_v_dim
QKVZ_DIM = 8
BA_DIM = 2 * HV
SENTINEL = 7.0

# Real-kernel dims; head_k_dim/head_v_dim=128 is what the decode kernel expects.
KERNEL_H = 4  # num key heads
KERNEL_HV = 8  # num value heads
KERNEL_K = 128  # head_k_dim
KERNEL_V = 128  # head_v_dim
CONV_KERNEL = 4
CONV_DIM = 2 * KERNEL_H * KERNEL_K + KERNEL_HV * KERNEL_V

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the packed decode kernel needs a GPU",
)


class _ReachedGenericPath(Exception):
    """Marks ``_forward_core`` continuing past the packed decode dispatch."""


def _make_metadata(
    *,
    num_actual_tokens: int,
    num_prefills: int = 0,
    num_decodes: int = 4,
    spec: bool = False,
    non_spec_state_indices: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_actual_tokens if num_prefills else 0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=1 if spec else 0,
        num_spec_decode_tokens=1 if spec else 0,
        num_actual_tokens=num_actual_tokens,
        non_spec_state_indices_tensor=non_spec_state_indices,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool) if spec else None,
    )


def _build_layer(*, packed_decode: bool) -> types.SimpleNamespace:
    """A minimal layer running the real ``_forward_core_rocm`` fill logic."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = packed_decode
    # False keeps _forward_core_rocm off the AITER decode branch, which returns
    # before the fill, so the fill logic under test is reached.
    layer.gqa_interleaved_layout = False
    layer.core_attn_out_at_forward_core = None

    def _prepare_inputs(qkvz, ba, num_tokens_all):
        z = torch.zeros(num_tokens_all, HV, V)
        return qkvz, z, ba, ba

    def _forward_core(mixed_qkv, b, a, core_attn_out):
        # Record what the fill left behind; the real kernels are irrelevant here.
        layer.core_attn_out_at_forward_core = core_attn_out.clone()

    layer.prepare_gdn_attention_core_inputs = _prepare_inputs
    layer._forward_core = _forward_core
    for name in ("_forward_core_rocm", "_use_packed_recurrent_decode"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _run_forward_core_rocm(layer, meta, num_tokens: int) -> torch.Tensor:
    """Run the wrapper, returning the buffer it handed to ``_forward_core``.

    The buffer records the decision by itself: the fill is all-or-nothing, so a
    buffer still holding the sentinel is one no fill ran on, which at runtime is
    the kernel launch the fast path exists to avoid.
    """
    # torch.empty semantics at the call site: pre-fill with a sentinel so an
    # unwritten row is distinguishable from a zeroed one.
    core_attn_out = torch.full((num_tokens, HV, V), SENTINEL)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})

    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, QKVZ_DIM),
            ba=torch.zeros(num_tokens, BA_DIM),
            z_out=torch.zeros(num_tokens, HV, V),
            core_attn_out=core_attn_out,
        )
    assert layer.core_attn_out_at_forward_core is not None
    return layer.core_attn_out_at_forward_core


def test_pure_decode_runs_no_fill_when_unpadded() -> None:
    """Unpadded pure decode: the packed kernel owns every row, so no fill runs.

    This is the case the fast path is for -- no launch at all, not a cheaper
    one -- and it is what an unpadded batch (eager, or a full decode cudagraph
    whose num_actual_tokens is itself token-padded) produces.
    """
    num_tokens = 4
    layer = _build_layer(packed_decode=True)
    meta = _make_metadata(num_actual_tokens=num_tokens, num_decodes=num_tokens)

    seen = _run_forward_core_rocm(layer, meta, num_tokens)

    assert torch.equal(seen, torch.full_like(seen, SENTINEL))


def test_padded_pure_decode_keeps_the_full_fill() -> None:
    """Padded pure decode: rows past ``num_actual_tokens`` are nobody's, so the
    fill stays."""
    num_tokens, num_actual_tokens = 6, 4
    layer = _build_layer(packed_decode=True)
    meta = _make_metadata(
        num_actual_tokens=num_actual_tokens, num_decodes=num_actual_tokens
    )

    seen = _run_forward_core_rocm(layer, meta, num_tokens)

    assert torch.equal(seen, torch.zeros_like(seen))


@pytest.mark.parametrize(
    "packed_decode,kwargs",
    [
        # The generic path may leave rows unwritten, so all of these must keep
        # the full fill that protects the torch.empty buffer.
        (True, {"num_prefills": 1, "num_decodes": 0}),
        (True, {"num_prefills": 1, "num_decodes": 2}),
        (True, {"num_decodes": 0}),
        (True, {"spec": True}),
        (False, {}),
    ],
)
def test_batches_that_miss_the_fast_path_keep_the_full_fill(
    packed_decode: bool, kwargs: dict
) -> None:
    num_tokens = 4
    layer = _build_layer(packed_decode=packed_decode)
    meta = _make_metadata(num_actual_tokens=num_tokens, **kwargs)

    seen = _run_forward_core_rocm(layer, meta, num_tokens)

    assert torch.equal(seen, torch.zeros_like(seen))


@pytest.mark.parametrize(
    "packed_decode,kwargs",
    [
        (True, {}),
        (True, {"num_prefills": 1, "num_decodes": 0}),
        (True, {"num_prefills": 1, "num_decodes": 2}),
        (True, {"num_decodes": 0}),
        (True, {"spec": True}),
        (False, {}),
    ],
)
def test_predicate_matches_forward_core_dispatch(
    packed_decode: bool, kwargs: dict
) -> None:
    """The fill decision must track the dispatch it is derived from.

    Skipping the fill is only safe when ``_forward_core`` actually reaches the
    packed decode kernel, so drift between the two would silently expose the
    uninitialised buffer.
    """
    num_tokens = 4
    meta = _make_metadata(num_actual_tokens=num_tokens, **kwargs)

    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = packed_decode
    layer.kv_cache = (torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))
    layer.conv1d = types.SimpleNamespace(
        weight=torch.zeros(1, 1, 1), bias=torch.zeros(1)
    )
    layer.activation = "silu"
    dispatched: list[bool] = []

    def _decode_non_spec(**_):
        dispatched.append(True)

    layer._forward_core_decode_non_spec = _decode_non_spec
    for name in ("_forward_core", "_use_packed_recurrent_decode"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )

    expected = layer._use_packed_recurrent_decode(meta)

    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})

    def _raise(*_, **__):
        raise _ReachedGenericPath

    # The generic path's first module-level call after the dispatch, so it
    # halts execution before any kernel without needing full metadata.
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx),
        patch.object(qwen_gdn_linear_attn, "is_conv_state_dim_first", _raise),
    ):
        try:
            layer._forward_core(
                mixed_qkv=torch.zeros(num_tokens, QKVZ_DIM),
                b=torch.zeros(num_tokens, HV),
                a=torch.zeros(num_tokens, HV),
                core_attn_out=torch.zeros(num_tokens, HV, V),
            )
            reached_generic = False
        except _ReachedGenericPath:
            reached_generic = True

    assert bool(dispatched) is expected
    assert reached_generic is not expected


def _build_kernel_layer(device: torch.device, pool_size: int):
    """A minimal layer running the real ``_forward_core_decode_non_spec``."""
    conv_state_shape, ssm_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, KERNEL_H, KERNEL_HV, KERNEL_K, KERNEL_V, CONV_KERNEL, num_spec=0
        )
    )
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.tp_size = 1
    layer.num_k_heads = KERNEL_H
    layer.num_v_heads = KERNEL_HV
    layer.head_k_dim = KERNEL_K
    layer.head_v_dim = KERNEL_V
    layer.activation = "silu"
    layer.A_log = torch.randn(KERNEL_HV, dtype=torch.float32, device=device) * 0.1
    layer.dt_bias = torch.randn(KERNEL_HV, dtype=torch.float32, device=device) * 0.1
    layer.conv1d = types.SimpleNamespace(
        weight=torch.randn(
            CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
        )
        * 0.1,
        bias=torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
    )
    layer.kv_cache = (
        torch.randn(pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device)
        * 0.05,
        torch.randn(pool_size, *ssm_state_shape, dtype=torch.float32, device=device)
        * 0.05,
    )
    layer._forward_core_decode_non_spec = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_decode_non_spec, layer
    )
    return layer


@requires_gpu
@pytest.mark.parametrize("num_actual_tokens", [1, 4, 7])
def test_packed_decode_kernel_writes_every_row_it_owns(num_actual_tokens: int) -> None:
    """The invariant that lets the wrapper skip the fill.

    Skipping the fill is only safe because the packed decode kernel leaves no
    row of ``core_attn_out[:num_actual_tokens]`` at its prior value. Poisoning
    the buffer with NaN makes any unwritten row observable, which a zero-filled
    buffer would hide.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    # A padding tail the kernel must not be expected to touch.
    num_tokens = num_actual_tokens + 2
    pool_size = num_actual_tokens + 1

    layer = _build_kernel_layer(device, pool_size)
    meta = _make_metadata(
        num_actual_tokens=num_actual_tokens,
        num_decodes=num_actual_tokens,
        non_spec_state_indices=torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ).clamp_(max=pool_size - 1),
    )

    core_attn_out = torch.full(
        (num_tokens, KERNEL_HV, KERNEL_V),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    layer._forward_core_decode_non_spec(
        mixed_qkv=torch.randn(num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device)
        * 0.1,
        b=torch.randn(num_tokens, KERNEL_HV, dtype=torch.bfloat16, device=device) * 0.1,
        a=torch.randn(num_tokens, KERNEL_HV, dtype=torch.bfloat16, device=device) * 0.1,
        core_attn_out=core_attn_out,
        attn_metadata=meta,
    )

    owned = core_attn_out[:num_actual_tokens]
    assert not owned.isnan().any(), (
        "packed decode kernel left rows of core_attn_out[:num_actual_tokens] "
        "unwritten; the wrapper's fill elimination would expose them"
    )
    # The tail is the wrapper's responsibility, not the kernel's.
    assert core_attn_out[num_actual_tokens:].isnan().all()
