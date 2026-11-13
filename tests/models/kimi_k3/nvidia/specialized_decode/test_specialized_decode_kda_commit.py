# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FP32-state RecoverSSM commit (`KdaCommitF32`, the `fp32_deferred` backend
`KDARecoverSSMCommitContext` gets under the specialized decode for FP32 checkpoints
on SM10x) against the stock
Triton commit it replaces: every layer's final state and, with prefix caching
(align), the block-boundary state, over random accepted lengths."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMRecords,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (  # noqa: E402
    KDADeferredCommitContext,
)

D, LOWER_BOUND = 128, -5.0
DEVICE = torch.device("cuda")


def _layers(heads, requests, q, slots, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    layers = []
    for _ in range(2):
        records = KDARecoverSSMRecords.allocate(
            requests, heads, q, D, torch.bfloat16, DEVICE
        )
        records.correction.copy_(randn(*records.correction.shape))
        records.key_gate.copy_(randn(*records.key_gate.shape).bfloat16())
        conv = randn(slots, 3 * heads * D, q + 2).bfloat16()
        layers.append(
            SimpleNamespace(
                kv_cache=(conv, randn(slots, heads, D, D) * 0.1),
                recoverssm_records=records,
                A_log=randn(heads) * 0.5,
                dt_bias=randn(heads * D) * 0.5,
                local_num_heads=heads,
                head_dim=D,
                gate_lower_bound=LOWER_BOUND,
            )
        )
    return layers


def _commit(layers, backend, q, requests, accepted, sources, align):
    context = KDADeferredCommitContext.create(
        layers, spec_query_len=q, max_num_reqs=requests, backend="fp32_deferred"
    )
    context.backend = backend
    starts = torch.arange(0, (requests + 1) * q, q, dtype=torch.int32, device=DEVICE)
    kwargs = {}
    if align is not None:
        table, computed, block = align
        kwargs = dict(block_table=table, num_computed_tokens=computed)
        kwargs.update(mamba_block_size=block)
    context.commit(accepted, sources, starts, **kwargs)


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("heads,requests,q", [(12, 9, 5), (96, 3, 8), (12, 4, 2)])
@torch.inference_mode()
def test_commit_matches_stock(heads, requests, q, align, monkeypatch):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import commit

    monkeypatch.setattr(commit, "COMMIT_ITEMS", 0)  # at every batch
    seed = heads + requests + q
    blocks = 3  # align: each request owns 3 state blocks
    slots = 1 + requests * blocks
    layers = _layers(heads, requests, q, slots, seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    accepted = torch.randint(1, q + 1, (requests,), device=DEVICE, generator=g)
    accepted = accepted.int()
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    table = table.view(requests, blocks)
    sources = table[:, 0].contiguous()
    spec = None
    if align:
        block = 8
        # Computed tokens just before a block boundary: most windows cross it.
        computed = torch.randint(1, block, (requests,), device=DEVICE, generator=g)
        computed = (computed + block - q // 2).int()
        spec = (table, computed, block)
    stock = [tuple(t.clone() for t in layer.kv_cache) for layer in layers]
    ours = [tuple(t.clone() for t in layer.kv_cache) for layer in layers]
    for caches, backend in ((stock, "triton"), (ours, "fp32_deferred")):
        run = [
            SimpleNamespace(**{**vars(layer), "kv_cache": cache})
            for layer, cache in zip(layers, caches)
        ]
        _commit(run, backend, q, requests, accepted, sources, spec)
    for layer, (stock_conv, stock_state), (conv, state) in zip(layers, stock, ours):
        torch.testing.assert_close(conv, stock_conv, atol=0, rtol=0)
        # FP32 rounding of a few products (the shared-key products are hoisted).
        torch.testing.assert_close(state, stock_state, atol=1e-6, rtol=1e-5)
        moved = (stock_state != layer.kv_cache[1]).flatten(1).any(1).sum()
        assert moved >= requests, moved  # every final slot (and some boundaries)


@torch.inference_mode()
def test_unbounded_gate_falls_back_to_triton():
    """The softplus gate (`gate_lower_bound=None`) keeps the stock Triton commit."""
    layers = _layers(12, 2, 5, 3, seed=0)
    for layer in layers:
        layer.gate_lower_bound = None
    context = KDADeferredCommitContext.create(
        layers, spec_query_len=5, max_num_reqs=2, backend="fp32_deferred"
    )
    assert context.backend == "triton"


def test_commit_policy_counts_layers():
    """The commit folds every KDA layer in one launch: KdaCommitF32 from 384 (head,
    request, layer) items, i.e. at every batch for a 69-layer rank."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import commit

    assert commit.use_commit_f32(12, 1, 69)
    assert not commit.use_commit_f32(12, 1, 1)
