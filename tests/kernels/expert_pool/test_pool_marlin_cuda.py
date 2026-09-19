# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert pool through the real Marlin consumer (CUDA).

A layer with a small ``moe_expert_pool_rows`` resident count: decode steps
(one token) must match the uncached layer while experts are promoted and
evicted through the shared bank (bank rows > expert count, so the
logical-align + physical-remap path is exercised), and a wider batch must
match through the bank + host-view partition path. The pool tables must
stay consistent throughout."""

import pytest
import torch

from tests.kernels.expert_pool.marlin_fixture import (
    TOP_K,
    E,
    K,
    dist_env,  # noqa: F401
    make_layer,
    quantized_weights,
    routing,
    vllm_config,
)
from vllm.model_executor.layers.fused_moe.expert_pool.install import (
    install_expert_pool,
)
from vllm.model_executor.layers.fused_moe.expert_pool.pool import verify_bank_rows
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    check_global_tables,
    resident_per_layer,
    set_gate,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
    pytest.mark.skipif(
        current_platform.is_cuda() and not is_fp4_marlin_supported(),
        reason="FP4 Marlin not supported on this GPU",
    ),
]

SLOTS = 4  # of E=8 experts resident per layer at start; top_k=2 staging rows


def _decode(order, device):
    logits = torch.full((1, E), -10.0, device=device)
    logits[0, order[0]] = 3.0
    logits[0, order[1]] = 2.0
    return logits


def test_two_layer_pool_decode_prefill_decode_matches_the_uncached_layers(
    dist_env,  # noqa: F811
):
    """Two layers share one bank (2 * SLOTS + staging = 10 rows > E = 8), so
    every bank call takes the logical-align + physical-remap path and a miss
    on one layer can evict the other layer's row. Decode on both layers,
    then a wide batch (bank + host-view partitions), then decode again on
    the same pool; every output must match the uncached layer."""
    device = torch.accelerator.current_accelerator()
    # One config per layer and side: a layer registers in the static forward
    # context of the config it was built under, which set_forward_context
    # must see again at run time (the legacy lookup also resolves layers by
    # call order within a config, so two layers never share one).
    ref_cfgs, pool_cfgs, refs, layers = [], [], [], []
    for seed_offset in (0, 1):
        params = quantized_weights(device, seed_offset=seed_offset)
        ref_cfgs.append(vllm_config(0))
        pool_cfgs.append(vllm_config(SLOTS))
        refs.append(make_layer(ref_cfgs[-1], params))
        layers.append(make_layer(pool_cfgs[-1], params, host_source=True))
    # As after the real loader: the small per-expert globals stay on the
    # device (never allocated in host memory), the big tensors are pinned.
    for layer in layers:
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            p = getattr(layer.routed_experts, name)
            p.data = p.data.to(device)
        # A non-contiguous (strided) host source must be densified into the
        # pool's own pinned copy, values preserved, not stride-preserved.
        p = layer.routed_experts.w2_weight_scale
        wide = torch.zeros((p.shape[0], 2, *p.shape[1:]), dtype=p.dtype).pin_memory()
        wide[:, 0].copy_(p.data)
        strided_values = p.data.clone()
        p.data = wide[:, 0]
        assert not p.data.is_contiguous()
    model = torch.nn.ModuleDict({"a": layers[0], "b": layers[1]})
    pool = install_expert_pool(model, device, max_decode_tokens=1)
    assert pool is not None
    # The pool owns pinned host copies of every source; the initial bank rows
    # match them byte for byte.
    for pl in (layer.routed_experts.expert_pool_layer for layer in layers):
        assert all(
            t.device.type == "cpu" and t.is_pinned() and t.is_contiguous()
            for t in pl.sources.values()
        )
    src = layers[-1].routed_experts.expert_pool_layer.sources["w2_weight_scale"]
    assert src.shape == strided_values.shape and src.dtype == strided_values.dtype
    assert torch.equal(src, strided_values)
    report = verify_bank_rows(pool, model.expert_pool_sources, sample=SLOTS)
    assert report == {"rows_checked": 2 * SLOTS, "rows_resident": 2 * SLOTS}
    assert pool.rows == 2 * SLOTS + TOP_K and pool.rows > E
    assert resident_per_layer(pool.tables) == [SLOTS, SLOTS]
    pls = [layer.routed_experts.expert_pool_layer for layer in layers]
    assert all(pl is not None and pl.bank_rows == pool.rows for pl in pls)
    check_global_tables(pool.tables)
    from vllm.forward_context import set_forward_context

    def run(i, x, logits, n):
        with set_forward_context(None, ref_cfgs[i], num_tokens=n):
            want = refs[i](x, logits)
        with set_forward_context(None, pool_cfgs[i], num_tokens=n):
            got = layers[i](x, logits)
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        check_global_tables(pool.tables)
        assert int(pool.tables.error[0]) == 0

    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    tables = pool.tables
    # Gate closed: a miss is staged into a shared staging row (physical row
    # >= E) and the step map must point there; the output still matches.
    set_gate(tables, False)
    run(0, x, _decode([6, 7], device), 1)  # experts 6, 7 are not resident
    step_map = pls[0].buffers.step_map.cpu().tolist()
    assert step_map[6] >= E and step_map[7] >= E, step_map
    assert resident_per_layer(tables) == [SLOTS, SLOTS]  # placement untouched
    set_gate(tables, True)
    # Decode steps whose routes walk every expert of both layers: misses
    # promote (evicting the least recently used row of either layer) or
    # stage into the shared staging rows.
    for order in ([0, 1], [4, 5], [6, 7], [2, 3], [0, 6], [7, 1]):
        run(0, x, _decode(order, device), 1)
    hot0_before = tables.layer_slice(tables.hot_phys, 0).cpu().clone()
    row_key_before = tables.row_key.cpu().clone()
    for order in ([4, 5], [6, 7], [2, 6]):
        run(1, x, _decode(order, device), 1)
    # Cross-layer eviction: layer 1's misses took rows from layer 0.
    hot0_after = tables.layer_slice(tables.hot_phys, 0).cpu()
    assert not torch.equal(hot0_before, hot0_after)
    row_key_after = tables.row_key.cpu()
    changed = (row_key_before != row_key_after).nonzero().flatten().tolist()
    assert changed and all(
        int(row_key_before[r]) // E == 0 and int(row_key_after[r]) // E == 1
        for r in changed
    ), (row_key_before.tolist(), row_key_after.tolist())
    assert sum(resident_per_layer(tables)) == 2 * SLOTS
    # Wide batch on layer 0: resident rows from the bank, the rest through
    # the host view; every route covered exactly once.
    xb = torch.randn(8, K, dtype=torch.bfloat16, device=device)
    run(0, xb, routing(list(range(E)), device), 8)
    assert pls[0].partition_steps == 1
    # Decode again on the same pool after the wide batch.
    for order in ([3, 4], [7, 0]):
        run(0, x, _decode(order, device), 1)
    run(1, x, _decode([0, 1], device), 1)
    assert pls[0].decode_steps == 9 and pls[1].decode_steps == 4  # 1 gate-closed
