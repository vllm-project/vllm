# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark specialized decode's router, routing and package GEMMs."""

import pytest
import torch

from vllm.platforms import current_platform

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the Muse Spark specialized decode's router needs an SM100-family GPU",
)
SENTINEL = 0x5A5A5A5A


@requires_sm100
@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize(
    "policy,n,k",
    [("UNPOOLER", 1024, 4096), ("POOLER", 512, 8192), ("OPROJ", 8192, 1024)],
)
@torch.inference_mode()
def test_tp8_package_gemm_matches_fp32_reference(policy, n, k, rows):
    """The unpooler, the cluster split-K pooler and the o_proj TP8 shards
    round once from the FP32 sum."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.common import (
        policy as policies,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.gemm.unpooler import (  # noqa: E501
        Unpooler,
    )

    torch.manual_seed(rows)
    weight = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    x = (torch.randn(rows, k, device="cuda") * 0.5).bfloat16()
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    Unpooler(weight, shape=getattr(policies, policy)).run(x, out)
    reference = x.float() @ weight.float().T
    torch.testing.assert_close(out.float(), reference, rtol=2**-8, atol=1e-3)


def _routing_logits(case: str, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(tokens, 256, device="cuda") * 3
    bias = torch.randn(256, device="cuda") * 0.1
    if case in ("ties", "equal", "adjacent_floats_at_the_boundary"):
        bias.zero_()
    if case == "ties":
        logits = torch.round(logits)
    elif case == "equal":
        logits.zero_()
    elif case == "infinite":
        logits[:, 7] = float("inf")
        logits[:, 9] = float("-inf")
    elif case == "adjacent_floats_at_the_boundary":
        for row in logits:
            order = row.argsort(descending=True)
            row[order[8]] = torch.nextafter(row[order[7]], row.new_tensor(-1e30))
    return logits, bias


def _expected_metadata(ids, offset, num_local, tile, size):
    """The metadata from its definition: one CTA per tile of an expert's
    tokens in expert order, rows in token order, -1 padding."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe import routing

    tokens_of: list[list[int]] = [[] for _ in range(num_local)]
    for token, row in enumerate(ids.tolist()):
        for expert in row:
            if 0 <= expert - offset < num_local:
                tokens_of[expert - offset].append(token)
    meta, ctas, first_cta = [SENTINEL] * size, 0, {}
    for expert, tokens in enumerate(tokens_of):
        if not tokens:
            continue
        first_cta[expert] = ctas
        n = -(-len(tokens) // tile)
        for j in range(n):
            meta[routing.META_CTA_BATCH + ctas + j] = expert
            meta[routing.META_CTA_LIMIT + ctas + j] = min(
                (ctas + j + 1) * tile, ctas * tile + len(tokens)
            )
        for rank in range(n * tile):
            meta[routing.META_PERMUTED_TO_TOKEN + ctas * tile + rank] = (
                tokens[rank] if rank < len(tokens) else -1
            )
        ctas += n
    meta[routing.META_PADDED_TOKENS] = ctas * tile
    meta[routing.META_NUM_CTAS] = ctas
    for token, row in enumerate(ids.tolist()):
        for k, expert in enumerate(row):
            local = expert - offset
            meta[routing.META_EXPANDED_TO_PERMUTED + token * 8 + k] = (
                first_cta[local] * tile + tokens_of[local].index(token)
                if 0 <= local < num_local
                else -1
            )
    return torch.tensor(meta, dtype=torch.int32)


@requires_sm100
@pytest.mark.parametrize("tokens", range(1, 9))
@pytest.mark.parametrize("offset,num_local,tile", [(0, 256, 8), (17, 5, 3)])
@pytest.mark.parametrize(
    "case", ["random", "ties", "equal", "infinite", "adjacent_floats_at_the_boundary"]
)
@torch.inference_mode()
def test_cute_routing_matches_fused_routing(tokens, offset, num_local, tile, case):
    """Ids and weights are bitwise the fused Muse Spark routing's (ties to the
    lowest expert) and the whole metadata buffer, guard and untouched elements
    included, is the reference's."""
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_muse_spark_routing,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe import routing

    torch.manual_seed(tokens * 1000 + offset + tile)
    logits, bias = _routing_logits(case, tokens)
    ref_weights, ref_ids = fused_muse_spark_routing(
        logits, bias, 8, 1.5, router_score_division_eps=1e-15
    )
    ids = torch.full((tokens, 8), -7, dtype=torch.int32, device="cuda")
    weights = torch.full((tokens, 8), float("nan"), device="cuda")
    size = routing.metadata_size(tile)
    meta = torch.full((size,), SENTINEL, dtype=torch.int32, device="cuda")
    routing.prepare()
    routing.route(
        logits,
        bias,
        ids,
        weights,
        1.5,
        metadata=meta,
        local_expert_offset=offset,
        num_local_experts=num_local,
        tile_n=tile,
    )
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    expected = _expected_metadata(ids.cpu(), offset, num_local, tile, size)
    assert torch.equal(meta.cpu(), expected)


@requires_sm100
@pytest.mark.parametrize("m", range(1, 9))
@torch.inference_mode()
def test_mma_router_partials_route_as_the_fp32_router(m):
    """The MMA router's K-split partials sum to within 1e-5 relative L2 of the
    FP64 logits, and routing them (`splits`) picks the experts and weights of
    routing the FP32 router kernel's logits (as split 0 of zero partials)."""
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp32w import (
        LL_FP32W_GEMM_KERNEL,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe import routing
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe.router_mma import (  # noqa: E501
        SPLITS,
        MmaRouter,
    )

    torch.manual_seed(m)
    w = torch.randn(256, 8192, device="cuda") * 0.02
    x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
    router = MmaRouter(w)
    partials = router.partials(x)
    logits = router.split_partials()[:, :m].double().sum(0)
    ref = x.double() @ w.double().t()
    assert ((logits - ref).norm() / ref.norm()).item() < 1e-5

    bias = torch.randn(256, device="cuda") * 0.1
    fp32_logits = torch.zeros_like(router.split_partials())
    fp32_logits[0, :m] = LL_FP32W_GEMM_KERNEL(x, w)
    routing.prepare(splits=SPLITS)
    selections = []
    for source in (partials, fp32_logits[0, :m]):
        ids = torch.empty(m, 8, device="cuda", dtype=torch.int32)
        weights = torch.empty(m, 8, device="cuda")
        routing.route(source, bias, ids, weights, 2.5, splits=SPLITS)
        order = ids.sort(-1)
        selections.append((order.values, weights.gather(-1, order.indices)))
    assert torch.equal(selections[0][0], selections[1][0])
    torch.testing.assert_close(selections[0][1], selections[1][1], rtol=1e-5, atol=0)
