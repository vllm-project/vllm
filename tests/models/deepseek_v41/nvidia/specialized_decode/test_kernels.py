# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU kernels of the DeepSeek-V4.1 specialized decode
(``vllm/models/deepseek_v41/nvidia/specialized_decode``), each against the stock path
it replaces, eagerly and replayed from a CUDA graph where the decode captures
it. Needs one SM100-family GPU; about 5 minutes, mostly kernel compilation.

Run: pytest -v tests/models/deepseek_v41/nvidia/specialized_decode/test_kernels.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.common.engram import Engram
from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
    indexer_k_norm_rope_store,
)
from vllm.models.deepseek_v41.nvidia import flashinfer_sparse
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
    mxfp8 as mxfp8_linear,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import sparse_mla
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.aux_projections import (  # noqa: E501
    aux_projection_gemm,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk import (  # noqa: E501
    select_candidate_blocks_cooperative,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.indexer_k_store import (  # noqa: E501
    indexer_k_norm_rope_store_fp32,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.sparse_mla import (
    SparseIndexTables,
    SparseMlaDecode,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.engram.prefetch import (
    EngramPrefetch,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    EXPERTS,
    HIDDEN,
    PADDED,
    SLICE,
    TILE,
    TOP_K,
    FrontConfig,
    RoutedExpertGemm,
    cta_words,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.metadata_graphs import (
    AttnMetadataCudaGraphs,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="needs an SM100-family GPU",
)

DEVICE = torch.device("cuda")


def _capture(fn):
    """``fn`` run once eagerly (compiling its kernels), then captured into a
    CUDA graph; returns the graph and the captured outputs."""
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in BF16 ulps between two BF16 tensors."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(a) - ordered(b)).abs()


# --------------------------------------------------------- routed experts


def _routing(
    rows: int, generator: torch.Generator, pool_size: int = 22
) -> torch.Tensor:
    """Top-k ids of ``rows`` rows drawn from ``pool_size`` experts (22, as a
    decode step of six rows routes); above eight rows experts own several
    tiles, and with a pool of six every expert takes every row."""
    pool = torch.randperm(EXPERTS, generator=generator)[:pool_size]
    ids = [
        pool[torch.randperm(pool_size, generator=generator)[:TOP_K]]
        for _ in range(rows)
    ]
    return torch.stack(ids).to(torch.int32).to(DEVICE)


@pytest.fixture(scope="module")
def experts():
    """Random MXFP4 experts converted by vLLM for the TRT-LLM-gen backend, the
    tensors both the FlashInfer path and the low-batch kernels read."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_weight_to_mxfp4_moe_kernel_format,
    )

    generator = torch.Generator().manual_seed(0)
    e, p, h = EXPERTS, PADDED, HIDDEN

    def randint(low, high, *shape):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=generator)

    w13, w13_scale = (
        randint(0, 256, e, 2 * p, h // 2),
        randint(118, 126, e, 2 * p, h // 32),
    )
    w2, w2_scale = randint(0, 256, e, h, p // 2), randint(118, 126, e, h, p // 32)
    # The intermediate is 576 wide, padded with zeros to 640.
    w13.view(e, 2, p, -1)[:, :, SLICE:] = 0
    w2[:, :, SLICE // 2 :] = 0
    w13, w2, w13_scale, w2_scale, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        nn.Module(),
        w13.to(DEVICE),
        w2.to(DEVICE),
        w13_scale.to(DEVICE),
        w2_scale.to(DEVICE),
        None,
        None,
        _cache_permute_indices={},
    )
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device=DEVICE)
    return w13, w13_scale, w2, w2_scale, clamp


def _gemm(experts) -> RoutedExpertGemm:
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_mxfp8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    w13, w13_scale, w2, w2_scale, clamp = experts
    quant_config = mxfp4_mxfp8_moe_quant_config(
        w1_scale=w13_scale, w2_scale=w2_scale, is_scale_swizzled=False
    )
    return RoutedExpertGemm(
        w13,
        w13_scale,
        w2,
        w2_scale,
        clamp,
        TOP_K,
        MoEPrepareAndFinalizeNoDPEPModular(),
        quant_config,
    )


def _flashinfer_experts(experts, routing, output=None):
    """``TrtLlmMxfp4ExpertsModular``'s FlashInfer call on the same quantized
    input: GEMM2 rows and their permutation, or the finalized ``output``."""
    from flashinfer import trtllm_fp4_block_scale_routed_moe

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    w13, w13_scale, w2, w2_scale, clamp = experts
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(routing.topk_ids, routing.topk_weights),
        routing_bias=None,
        hidden_states=routing.x_quant,
        hidden_states_scale=routing.x_scale.view(torch.float8_e4m3fn),
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=PADDED,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        do_finalize=output is not None,
        enable_pdl=True,
        activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
        output=output,
        tune_max_num_tokens=max(routing.topk_ids.size(0), 8),
    )
    if output is not None:
        return output
    rows, _, permuted = result
    return rows[permuted.view(routing.topk_ids.shape).long()]


def _fma_finalize(rows: torch.Tensor, permuted: torch.Tensor, weights: torch.Tensor):
    """The top-k reduction as the fused all-reduce + mHC kernel computes it: an
    FP32 fused multiply-add chain over k (emulated in FP64), then BF16."""
    rows, permuted, weights = (
        rows.cpu().double(),
        permuted.cpu(),
        weights.cpu().double(),
    )
    out = torch.zeros(permuted.shape[0], HIDDEN, dtype=torch.float32)
    for k in range(permuted.shape[1]):
        out = (
            out.double() + weights[:, k : k + 1] * rows[permuted[:, k].long()]
        ).float()
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize(
    ("rows", "pool_size"),
    # Both sides of every GEMM width (8, 16, 32, 64, 128, 192 rows), one DSpark
    # request (6 rows), and crowded routing (6 experts) at the larger widths.
    [(r, 22) for r in (1, 6, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 171, 192)]
    + [(64, 6), (128, 6), (192, 6)],
)
def test_routed_experts_match_flashinfer(experts, rows, pool_size, graph):
    """Route + GEMM pair: the routed rows bitwise those of FlashInfer's
    TRT-LLM-gen MoE, and their top-k reduction within one BF16 ulp of its
    finalized output."""
    generator = torch.Generator().manual_seed(rows)
    gemm = _gemm(experts)

    def inputs():
        x = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
        weights = torch.softmax(torch.randn(rows, TOP_K, generator=generator), -1)
        ids = _routing(rows, generator, pool_size)
        return x.to(DEVICE), ids, weights.to(DEVICE)

    x, ids, weights = inputs()

    def run():
        routing = gemm.route(x, ids, weights)
        return routing, gemm.gemms(routing)

    if graph:
        cuda_graph, (routing, out) = _capture(run)
        for tensor, fresh in zip((x, ids, weights), inputs()):
            tensor.copy_(fresh)
        cuda_graph.replay()
    else:
        routing, out = run()
    routed = out.gemm2_permuted[out.expanded_idx_to_permuted_idx.long()]
    assert torch.equal(routed, _flashinfer_experts(experts, routing))

    finalized = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    _flashinfer_experts(experts, routing, finalized)
    ours = _fma_finalize(
        out.gemm2_permuted, out.expanded_idx_to_permuted_idx, out.expert_weights
    )
    assert _bf16_ulps(ours, finalized.cpu()).max() <= 1


@pytest.mark.parametrize("rows", [6, 12, 48, 192])
def test_routed_experts_replays_agree(experts, rows):
    """GEMM2 waits for each token tile's GEMM1 items (the record's ready
    words), not for the GEMM1 grid. One captured route + GEMM pair is
    replayed 500 times on two alternating input sets (different rows and
    experts), and each replay's routed rows must be bitwise
    FlashInfer's for its set: a tile read before its H is complete would
    hold the other set's H. The used tiles' ready words must count GEMM1's
    m tiles. Six and twelve rows leave GEMM1 a partial last wave, as a
    decode step does."""
    generator = torch.Generator().manual_seed(1000 + rows)
    gemm = _gemm(experts)

    def inputs():
        x = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
        weights = torch.softmax(torch.randn(rows, TOP_K, generator=generator), -1)
        ids = _routing(rows, generator, min(EXPERTS, TOP_K * rows))
        return x.to(DEVICE), ids, weights.to(DEVICE)

    sets = [inputs(), inputs()]
    x, ids, weights = (t.clone() for t in sets[0])

    def load(index):
        for tensor, value in zip((x, ids, weights), sets[index]):
            tensor.copy_(value)

    def run():
        routing = gemm.route(x, ids, weights)
        return routing, gemm.gemms(routing)

    graph, (routing, out) = _capture(run)
    wants = []
    for index in range(2):
        load(index)
        graph.replay()
        wants.append(_flashinfer_experts(experts, routing))
    for replay in range(500):
        load(replay % 2)
        graph.replay()
        routed = out.gemm2_permuted[out.expanded_idx_to_permuted_idx.long()]
        assert torch.equal(routed, wants[replay % 2]), replay
    ready = 4 + 2 * cta_words(routing.ctas) + TILE * routing.ctas
    tiles = int(routing.record[1])
    counts = routing.record[ready : ready + tiles]
    assert torch.all(counts == 2 * SLICE // 128), counts


SENTINEL = 128815  # First of five image sentinel token ids.


def _front_inputs(rows: int, generator: torch.Generator, image_rows: int):
    """Router logits with exact ties, NaN, and values past the softplus
    thresholds; BF16 rows with a zero, a tiny and a large MXFP8 block."""
    logits = 3 * torch.randn(rows, EXPERTS, generator=generator)
    logits[:, 10:20] = logits[:, 10:11]
    logits[0, 30], logits[0, 31] = 25.0, -100.0
    if rows > 1:
        logits[1, 40] = float("nan")
    x = torch.randn(rows, HIDDEN, generator=generator)
    x[0, :32] = 0
    x[0, 32:64] *= 1e-30
    x[-1, 64:96] *= 1e4
    ids = torch.randint(0, 100000, (rows,), generator=generator)
    ids[:image_rows] = SENTINEL + torch.arange(image_rows) % 5
    ids[image_rows : image_rows + 1] = SENTINEL + 5
    return logits.to(DEVICE), x.to(torch.bfloat16).to(DEVICE), ids.to(DEVICE)


def _chain(gemm: RoutedExpertGemm, x, logits, ids, is_padding=None):
    """What the front replaces: the DSv4 top-k (with the router's padding
    mask), then quantization and route."""
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    config = gemm.front_config
    weights, topk_ids = dsv4_topk(
        logits,
        config.bias,
        torch.int32,
        config.scale,
        is_padding=is_padding,
        input_ids=ids,
        bias_vl=config.bias_vl,
        image_sentinel_lo=config.image_sentinel_lo,
    )
    return gemm.route(x, topk_ids, weights)


def _assert_same_routing(got, want):
    for field in (
        "topk_ids",
        "topk_weights",
        "x_quant",
        "x_scale",
        "permuted",
        "record",
    ):
        a, b = getattr(got, field), getattr(want, field)
        assert a.shape == b.shape, field
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), field
    assert got.ctas == want.ctas


@pytest.mark.parametrize(
    ("image", "scale"), [(None, 1.0), (torch.int32, 1.5), (torch.int64, 1.0)]
)
# Both sides of each build width (8, 16, 32, 64, 128, 192 rows), which also
# changes the rows a selecting CTA holds, and one DSpark request (6 rows).
@pytest.mark.parametrize(
    "rows", [1, 6, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 171, 192]
)
def test_front_matches_the_chain(experts, rows, image, scale):
    """The fused front: ids (in tie order), FP32 weights, MXFP8 bytes and
    scales, permutation and routing record bitwise those of the chain, with
    and without CUDA graph padding rows."""
    generator = torch.Generator().manual_seed(rows)
    gemm = _gemm(experts)
    bias = 0.05 * torch.randn(EXPERTS, generator=generator)
    bias[10:20] = bias[10]
    gemm.front_config = FrontConfig(
        bias=bias.to(DEVICE),
        bias_vl=(0.05 * torch.randn(EXPERTS, generator=generator)).to(DEVICE)
        if image
        else None,
        image_sentinel_lo=SENTINEL if image else 0,
        scale=scale,
    )

    def padding(count):
        return torch.arange(rows, device=DEVICE) >= rows - count

    for trial in range(6):
        logits, x, ids = _front_inputs(rows, generator, image_rows=trial % (rows + 1))
        if image is not None:
            ids = ids.to(image)
        # Trials 4 and 5: the last row, then the last half of the rows, padding.
        mask = None if trial < 4 else padding(1 if trial == 4 else rows // 2)
        _assert_same_routing(
            gemm.front(x, logits, ids, mask), _chain(gemm, x, logits, ids, mask)
        )

    # Replayed from a CUDA graph on fresh logits, rows and padding.
    mask = padding(0)
    graph, routing = _capture(lambda: gemm.front(x, logits, ids, mask))
    fresh_logits, fresh_x, _ = _front_inputs(rows, generator, image_rows=0)
    logits.copy_(fresh_logits)
    x.copy_(fresh_x)
    mask.copy_(padding(rows // 3))
    graph.replay()
    _assert_same_routing(routing, _chain(gemm, x, logits, ids, mask))


# ------------------------------------------------- candidate selection


@pytest.mark.parametrize(
    ("ends", "k"),
    [
        # Long rows around the 16384-block register-path limit and a short one.
        ((16384 * 8, 16384 * 8 - 3, 16385 * 8 + 1, 3000 * 8 + 5, 20000 * 8, 1), 2048),
        ((0, 100 * 8 + 3, 2048 * 8, 512 * 8), 512),
        ((131072 * 8, 131070, 9000 * 8 + 2), 1024),
    ],
)
def test_cooperative_candidates_match_stock(workspace_init, ends, k):
    """Each row's candidate set equals the stock ``torch.topk`` selection,
    including the pinned newest block and -inf blocks left as -1."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )

    block, nblocks = 8, 131072
    rows = len(ends)
    logits = torch.empty(rows, nblocks * block, device=DEVICE)
    row_ends = torch.tensor(ends, dtype=torch.int32, device=DEVICE)

    def fill_logits(seed):
        torch.manual_seed(seed)
        logits.normal_()
        logits[0, : 50 * block] = float("-inf")
        end = ends[-1]
        logits[-1, (end - 1) // block * block : end] = -100.0  # lowest, but pinned

    def select(selector):
        # In-range stale ids: a slot the selection skips would leak one.
        out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
        selector(out)
        return out

    def stock():
        return select(
            lambda out: select_candidate_blocks(logits, None, row_ends, k, block, out)
        )

    def cooperative():
        return select(
            lambda out: select_candidate_blocks_cooperative(
                logits, row_ends, block, out
            )
        )

    def assert_same_sets(got, want):
        assert torch.equal(got.sort(dim=1).values, want.sort(dim=1).values)

    fill_logits(0)
    assert_same_sets(cooperative(), stock())
    graph, out = _capture(cooperative)
    fill_logits(1)
    graph.replay()
    assert_same_sets(out, stock())


# ------------------------------------------------------ sparse-MLA decode


@pytest.mark.parametrize("query_lens", [[1], [6], [6, 6, 2]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sparse_mla_persistent_kv_counters(dtype, query_lens):
    """The decode on the shared KV counters equals the decode on fresh ones
    bitwise, launch after launch, and leaves the counters zeroed."""
    if sparse_mla.kv_counter_buffer(1) is None:
        pytest.skip("FlashInfer is not the version the persistent counters need")
    torch.manual_seed(0)
    heads, head_dim, page, top_k = 64, 512, 64, 1024
    tokens = sum(query_lens)

    def pool(pages):
        return (torch.randn(pages, page, head_dim, device=DEVICE) * 0.5).to(dtype)

    swa, compressed = pool(64), pool(256)
    query = torch.randn(tokens, heads, head_dim, device=DEVICE).to(dtype)
    indices = torch.cat(
        [
            torch.randint(0, 64 * page, (tokens, 128)),
            torch.randint(0, 256 * page, (tokens, top_k)),
        ],
        -1,
    )
    lens = torch.randint(128 + top_k // 2, 128 + top_k + 1, (tokens,))
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    def run(kv_counters):
        out = torch.empty(tokens, heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
        sparse_mla.trtllm_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=swa,
            compressed_kv_cache=compressed,
            workspace_buffer=workspace,
            sparse_indices=indices.to(torch.int32).to(DEVICE),
            sparse_topk_lens=lens.to(torch.int32).to(DEVICE),
            seq_lens=torch.full(
                (len(query_lens),), 8192, dtype=torch.int32, device=DEVICE
            ),
            out=out,
            bmm1_scale=head_dim**-0.5,
            bmm2_scale=1.0,
            sinks=torch.linspace(-1, 1, heads, device=DEVICE),
            cum_seq_lens_q=torch.tensor([0, *query_lens], device=DEVICE).cumsum(
                0, dtype=torch.int32
            ),
            max_q_len=max(query_lens),
            kv_counters=kv_counters,
        )
        return out

    reference = run(None)
    counters = sparse_mla.kv_counter_buffer(len(query_lens) * heads)
    assert counters is not None
    for _ in range(3):
        assert torch.equal(run(counters), reference)
        assert not counters.any()


# ----------------------------------------------- low-batch dense GEMMs


class _Mxfp8Linear(nn.Module):
    """An MXFP8 linear with weights processed for the ``cute-dsl`` kernel."""

    def __init__(self, n: int, k: int):
        super().__init__()
        w = torch.randn(n, k, device=DEVICE) / k**0.5
        w_q, w_s = mxfp8_e4m3_quantize(
            w.to(torch.bfloat16), is_sf_swizzled_layout=False
        )
        self.weight = nn.Parameter(w_q, requires_grad=False)
        self.weight_scale = nn.Parameter(w_s.view(n, k // 32), requires_grad=False)
        kernel = FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearLayerConfig((n, k)))
        kernel.process_weights_after_loading(self)
        self.quant_method = SimpleNamespace(kernel=kernel)

    def forward(self, x):
        return self.quant_method.kernel.apply_weights(self, x)


def _quantized(rows: int, k: int) -> QuantizedActivation:
    """An input pre-quantized as the fused producers emit it (F8_128x4 scales)."""
    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    data, scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    return QuantizedActivation(
        data, scale.reshape(-1), torch.bfloat16, x.shape, kMxfp8Dynamic
    )


@pytest.mark.parametrize("rows", [1, 5, 6, 8])
@pytest.mark.parametrize("shape", list(mxfp8_linear.PINNED_TACTICS), ids=str)
def test_low_latency_mxfp8_mm(shape, rows):
    """``fused_wqa_wkv``/``wq_b``/``wo_b`` at their TP4 shapes on
    ``cutedsl_low_latency``, eagerly and replayed, within BF16 tolerance of the
    stock ``cute-dsl`` layer; a BF16 input is left to the stock layer."""
    if not mxfp8_linear._backend_supported():
        pytest.skip("FlashInfer mm_mxfp8 has no cutedsl_low_latency backend")
    n, k = shape
    linear = _Mxfp8Linear(n, k)
    assert mxfp8_linear.mxfp8_eligible(linear)
    qa = _quantized(rows, k)
    out = mxfp8_linear.low_latency_mm(linear, qa)
    assert out is not None
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    graph, out = _capture(lambda: mxfp8_linear.low_latency_mm(linear, qa))
    qa.data.copy_(_quantized(rows, k).data)
    graph.replay()
    assert torch.equal(out, mxfp8_linear.mxfp8_mm(linear, qa))
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    assert mxfp8_linear.low_latency_mm(linear, x) is None


@pytest.mark.parametrize("rows", [9, 12, 18, 30, 48, 72, 192, 288, 384])
@pytest.mark.parametrize("shape", list(mxfp8_linear.CUTE_DSL_TACTICS), ids=str)
def test_pinned_mxfp8_mm(shape, rows):
    """``fused_wqa_wkv``/``wq_b``/``wo_b`` at their TP4 shapes on ``cute-dsl``
    with the pinned tactic of the row count, eagerly and replayed, within BF16
    tolerance of the stock layer, for MXFP8 and BF16 inputs."""
    if not current_platform.is_device_capability(103):
        pytest.skip("the tactics are pinned on GB300")
    n, k = shape
    linear = _Mxfp8Linear(n, k)
    qa = _quantized(rows, k)
    out = mxfp8_linear.pinned_mm(linear, qa)
    assert out is not None
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    graph, out = _capture(lambda: mxfp8_linear.pinned_mm(linear, qa))
    qa.data.copy_(_quantized(rows, k).data)
    graph.replay()
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    out = mxfp8_linear.pinned_mm(linear, x)
    assert out is not None
    torch.testing.assert_close(out, linear(x), atol=2e-2, rtol=1.6e-2)


@pytest.mark.parametrize("rows", [9, 12, 16, 17, 24, 32, 33, 48])
@pytest.mark.parametrize("shape", list(mxfp8_linear.WIDE_TACTICS), ids=str)
def test_wide_low_latency_mxfp8_mm(shape, rows):
    """``fused_wqa_wkv``/``wq_b``/``wo_b`` at their TP4 shapes on the
    low-latency kernel with a 16-, 32- or 48-row activation tile, eagerly and
    replayed, within BF16 tolerance of the stock layer, for MXFP8 and BF16
    inputs; above ``WIDE_LOW_LATENCY_MAX_ROWS`` rows it does not run."""
    if not mxfp8_linear._wide_supported():
        pytest.skip("the wide tactics are pinned on GB300 and the checked FlashInfer")
    n, k = shape
    linear = _Mxfp8Linear(n, k)
    qa = _quantized(rows, k)
    out = mxfp8_linear.wide_mm(linear, qa)
    assert out is not None
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    graph, out = _capture(lambda: mxfp8_linear.wide_mm(linear, qa))
    qa.data.copy_(_quantized(rows, k).data)
    graph.replay()
    assert torch.equal(out, mxfp8_linear.wide_mm(linear, qa))
    torch.testing.assert_close(out, linear(qa), atol=2e-2, rtol=1.6e-2)

    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    out = mxfp8_linear.wide_mm(linear, x)
    assert out is not None
    torch.testing.assert_close(out, linear(x), atol=2e-2, rtol=1.6e-2)
    assert mxfp8_linear.wide_mm(linear, _quantized(49, k)) is None


def test_pinned_mxfp8_mm_rows():
    """Up to ``LOW_BATCH_MAX_ROWS`` rows, and for an unpinned shape, the
    stock layer runs."""
    n, k = next(iter(mxfp8_linear.CUTE_DSL_TACTICS))
    assert mxfp8_linear.pinned_mm(_Mxfp8Linear(n, k), _quantized(8, k)) is None
    assert mxfp8_linear.pinned_mm(_Mxfp8Linear(1024, k), _quantized(48, k)) is None


@pytest.mark.parametrize("rows", [1, 5, 6, 8, 16, 17])
def test_low_batch_lm_head(dist_init, default_vllm_config, rows):
    """The logits processor's LM head on CuTe DSL ``mm_bf16`` for 1-16 rows,
    within BF16 rounding of the FP32 product; 17 rows keep the default GEMM."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.deepseek_v41.nvidia.specialized_decode.logits.lm_head import (
        low_batch_lm_head_applies,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.logits.logits import (
        SpecializedDecodeLogitsProcessor,
    )

    vocab = 129280 // 4
    with torch.device(DEVICE):
        head = ParallelLMHead(vocab, HIDDEN, params_dtype=torch.bfloat16)
    head.weight.data.normal_(std=0.02)
    x = torch.randn(rows, HIDDEN, device=DEVICE).to(torch.bfloat16)
    assert low_batch_lm_head_applies(head, x) == (rows <= 16)
    logits = SpecializedDecodeLogitsProcessor(vocab)._apply_head(head, x, None)
    expected = x.float() @ head.weight[:vocab].float().T
    torch.testing.assert_close(
        logits[:, :vocab].float(), expected, rtol=2**-7, atol=2**-7
    )


@pytest.mark.parametrize("rows", [1, 5, 6, 8])
@pytest.mark.parametrize(
    ("n", "k"),
    # Compressor kv + score and kv only, indexer weights, indexer wk.
    [(1024, 5120), (512, 5120), (64, 5120), (128, 512)],
)
def test_aux_projection_gemm(n, k, rows):
    """FP32 ``x @ W.T`` that differs from the exact product by no more than
    any K-term summation order can."""
    if not ll_bf16.is_available():
        pytest.skip("CuTe DSL ll_bf16 is unavailable")
    torch.manual_seed(rows)
    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    w = (torch.randn(n, k, device=DEVICE) / k**0.5).to(torch.bfloat16)
    out = aux_projection_gemm(x, w)
    assert out.dtype == torch.float32 and out.shape == (rows, n)
    exact = x.double() @ w.double().T
    bound = k * 2**-24 * (x.double().abs() @ w.double().abs().T)
    assert ((out.double() - exact).abs() <= bound).all()


@pytest.mark.parametrize("ratio", [1, 2])
def test_fp32_index_key_store(ratio):
    """The index key store fed by ``wk``'s FP32 output writes the MXFP4 cache
    bytes the shared store writes for its BF16 output."""
    if not ll_bf16.is_available():
        pytest.skip("CuTe DSL ll_bf16 is unavailable")
    torch.manual_seed(ratio)
    rows, block_size = 8, 64
    latent = torch.randn(rows, 512, device=DEVICE).to(torch.bfloat16)
    wk = (torch.randn(128, 512, device=DEVICE) / 512**0.5).to(torch.bfloat16)
    k_pre = aux_projection_gemm(latent, wk)
    positions = torch.arange(1001, 1001 + rows, device=DEVICE)
    cos_sin = torch.randn(4096, 64, device=DEVICE)
    norm_weight = torch.randn(128, device=DEVICE).to(torch.bfloat16)
    slots = torch.randperm(4 * block_size, device=DEVICE)[:rows]
    slots[3] = -1
    caches = [
        torch.zeros(4, block_size, 64 + 4, dtype=torch.uint8, device=DEVICE)
        for _ in range(2)
    ]
    indexer_k_norm_rope_store_fp32(
        k_pre, positions, cos_sin, norm_weight, 1e-6, caches[0], slots, ratio
    )
    indexer_k_norm_rope_store(
        k_pre.to(torch.bfloat16),
        positions,
        cos_sin,
        norm_weight,
        1e-6,
        caches[1],
        slots,
        ratio,
        use_fp4_cache=True,
    )
    assert caches[0].any()
    assert torch.equal(caches[0], caches[1])


# --------------------------------------------------------------- Engram


@pytest.mark.parametrize("masked", [False, True])
def test_engram_wkv_prefetch_matches_inline(masked):
    """The Engram WKV started on a side stream right after its (slow) host
    lookup, replayed from a CUDA graph, gives the stock forward's output
    bitwise."""
    from tests.kernels.test_engram import _make_embedding

    hc_mult, dim, max_tokens = 4, 256, 16
    table = _make_embedding(cpu_offload=True)
    lookup = table.lookup

    def slow_lookup(indices, out, background=False):
        torch.cuda._sleep(1_000_000)
        lookup(indices, out, background=background)

    table.lookup = slow_lookup
    engram = Engram.__new__(Engram)
    nn.Module.__init__(engram)
    engram.embed_tokens = table
    engram.hc_mult, engram.dim = hc_mult, dim
    engram.eps, engram.clamp_value = 1e-6, 1e-6
    engram.use_sequence_parallel = False
    engram.wkv = _Mxfp8Linear((hc_mult + 1) * dim, table.n_hash_cols * table.dim)
    engram.q_weight = torch.randn(hc_mult, dim, device=DEVICE).to(torch.bfloat16)
    engram.k_weight = torch.randn(hc_mult, dim, device=DEVICE).to(torch.bfloat16)
    with torch.device(DEVICE):
        engram._init_staging(max_tokens, table.dim)
    prefetch = EngramPrefetch([engram], max_tokens)
    wkv_out, _ = prefetch._buffers[engram]

    for num_tokens in (1, 6, max_tokens):
        hashes = torch.randint(
            0,
            table.part_num_embeddings,
            (num_tokens, table.n_hash_cols),
            dtype=torch.int32,
            device=DEVICE,
        )
        hidden = torch.randn(num_tokens, hc_mult, dim, device=DEVICE).to(torch.bfloat16)
        mask = torch.rand(num_tokens, device=DEVICE) > 0.3 if masked else None

        def run(hashes=hashes, hidden=hidden, mask=mask, num_tokens=num_tokens):
            engram.prepare_embeddings(hashes)
            prefetch.start(num_tokens)
            out = prefetch.forward(engram, hidden, hashes, mask)
            prefetch.join()
            return out

        graph, out = _capture(run)
        for _ in range(2):
            hashes.random_(0, table.part_num_embeddings)
            hidden.normal_()
            wkv_out.fill_(float("nan"))
            graph.replay()
            assert not wkv_out[:num_tokens].isnan().any()
            engram.prepare_embeddings(hashes)
            assert torch.equal(out, engram(hidden, hashes, mask))


# ------------------------------------------- sparse-index side builds

GROUPS = 4  # SWA KV-cache groups; layer i is in group i % GROUPS.
# (compress_ratio, kv source, index source) per layer.
LAYOUT = (
    (0, None, None),
    (0, None, None),
    *((1, 2, 2),) * 6,
    *((1, 2, 8),) * 4,
    # Its first layer hits the earlier SWA-only tables, so nothing forks.
    (0, None, None),
    (0, None, None),
    (0, None, None),
)
INDEX_SOURCES = (2, 8)
KEY_A = "model.layers.2.attn:2:1"  # The table key of layers 2-7.
DECODES, TOKENS, WINDOW, SIDE_TOP_K = 2, 4, 128, 512
SWA_BLOCK, COMPRESSED_BLOCK = 64, 256


def _swa_prefix(layer_id: int) -> str:
    return f"model.layers.{layer_id}.attn.swa_cache"


class SideBuildStep:
    """One pure-decode step through the sparse-index lookups of ``LAYOUT``,
    with real table builds and inputs refreshed in place."""

    def __init__(self, seed: int):
        self.generator = torch.Generator(device=DEVICE).manual_seed(seed)
        self.topk = torch.zeros(TOKENS, SIDE_TOP_K, dtype=torch.int32, device=DEVICE)
        # The top-k indices each index source's indexer writes.
        self.sources = {i: torch.zeros_like(self.topk) for i in INDEX_SOURCES}
        self.swa_caches = [
            torch.zeros(8, SWA_BLOCK, 584, dtype=torch.uint8, device=DEVICE)
            for _ in range(GROUPS)
        ]
        self.compressed_cache = torch.zeros(
            16, COMPRESSED_BLOCK, 584, dtype=torch.uint8, device=DEVICE
        )

        def tensor(*values, dtype=torch.int32):
            return torch.tensor(values, dtype=dtype, device=DEVICE)

        self.groups = [
            SimpleNamespace(
                num_decodes=DECODES,
                num_prefills=0,
                num_decode_tokens=TOKENS,
                num_prefill_tokens=0,
                seq_lens=tensor(300, 500),
                query_start_loc=tensor(0, 2, 4),
                query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
                max_decode_query_len=TOKENS // DECODES,
                token_to_req_indices=tensor(0, 0, 1, 1),
                decode_swa_indices=torch.zeros_like(self.topk[:, :WINDOW]),
                decode_swa_width=WINDOW,
                flashinfer_decode_topk_lens=None,
                flashinfer_decode_seq_lens=None,
                block_table=torch.zeros(DECODES, 8, dtype=torch.int32, device=DEVICE),
                replay_start=tensor(0, 0),
                is_valid_token=tensor(1, 1, 1, 1, dtype=torch.bool),
                block_size=SWA_BLOCK,
                flashinfer_sparse_index_cache={},
            )
            for _ in range(GROUPS)
        ]
        self.compressed = SimpleNamespace(
            block_table=torch.zeros(DECODES, 16, dtype=torch.int32, device=DEVICE),
            block_size=COMPRESSED_BLOCK,
        )
        self.by_layer = {
            _swa_prefix(i): self.groups[i % GROUPS] for i in range(len(LAYOUT))
        }
        self.by_layer["model.layers.2.attn"] = self.compressed
        self.refresh()

    def layers(self) -> list[DeepseekV4FlashInferMLAAttention]:
        layers = []
        for layer_id, (ratio, kv_source, index_source) in enumerate(LAYOUT):
            layer = object.__new__(DeepseekV4FlashInferMLAAttention)
            layer.compress_ratio = ratio
            layer.window_size = WINDOW
            layer.topk_indices_buffer = self.topk
            layer.compressed_cache_prefix = (
                None if kv_source is None else f"model.layers.{kv_source}.attn"
            )
            layer.index_source_layer_id = index_source
            layer.swa_cache_layer = SimpleNamespace(
                prefix=_swa_prefix(layer_id),
                kv_cache=self.swa_caches[layer_id % GROUPS],
            )
            layers.append(layer)
        return layers

    def _fill(self, tensor, low, high):
        tensor.copy_(
            torch.randint(
                low, high, tensor.shape, generator=self.generator, device=DEVICE
            )
        )

    def refresh(self):
        for source in self.sources.values():
            self._fill(source, -1, 300)
        for group in self.groups:
            self._fill(group.decode_swa_indices, -1, 500)
            self._fill(group.block_table, 0, 8)
        self._fill(self.compressed.block_table, 0, 16)

    def clear_tables(self):
        """What a fresh metadata build does to each group's table cache."""
        for group in self.groups:
            group.flashinfer_sparse_index_cache.clear()

    def lookup_args(self, layer):
        """The arguments of ``layer``'s sparse-index build."""
        swa_only = layer.compress_ratio == 0
        return (
            None if swa_only else self.compressed_cache,
            layer.swa_cache_layer.kv_cache,
            self.by_layer[layer.swa_cache_layer.prefix],
            None if swa_only else self.compressed,
            swa_only,
        )

    def forward(self, layers, tables, mode, stop=None, join_at_end=True):
        """The layers' table lookups in model order, on ``tables`` (None: the
        stock build of each layer): each index source joins the side builds,
        then its indexer writes the top-k indices; the executor joins after
        its layers. Returns copies of each layer's table."""
        out = []
        context = ForwardContext({}, self.by_layer, {}, cudagraph_runtime_mode=mode)
        with override_forward_context(context):
            if tables is not None:
                tables.reset()
            for layer_id, layer in enumerate(layers[:stop]):
                if layer_id in INDEX_SOURCES:
                    if tables is not None:
                        tables.join()
                    self.topk.copy_(self.sources[layer_id])
                args = self.lookup_args(layer)
                if tables is None:
                    *_, indices, lens = layer._build_sparse_index_metadata(*args)
                else:
                    *_, indices, lens = tables.lookup(layer, *args)
                out.append((indices.clone(), lens.clone()))
            if tables is not None and join_at_end:
                tables.join()
        return out

    def serial_tables(self):
        """Every layer's table from builds on the main stream only."""
        self.clear_tables()
        tables = self.forward(self.layers(), None, CUDAGraphMode.NONE)
        self.clear_tables()
        return tables


@pytest.fixture
def side_build_step(monkeypatch):
    """A step whose layers share side-stream builds that run late: a missing
    wait reads a table before it is written, and a missing join reads top-k
    indices after the next index source overwrote them."""
    step = SideBuildStep(seed=0)
    layers = step.layers()
    tables = SparseIndexTables(layers)
    build = flashinfer_sparse.build_flashinfer_mixed_sparse_indices

    def slow_build(*args, **kwargs):
        if torch.cuda.current_stream() == tables.stream:
            torch.cuda._sleep(2_000_000)
        return build(*args, **kwargs)

    monkeypatch.setattr(
        flashinfer_sparse, "build_flashinfer_mixed_sparse_indices", slow_build
    )
    return step, layers, tables


def _assert_tables_equal(actual, expected):
    assert len(actual) == len(expected)
    for (indices, lens), (want_indices, want_lens) in zip(actual, expected):
        assert torch.equal(indices, want_indices) and torch.equal(lens, want_lens)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_sparse_index_side_builds_match_serial(side_build_step, graph):
    """Every layer's sparse-index table equals the serial build's, eagerly and
    replayed from a CUDA graph on refreshed inputs."""
    step, layers, tables = side_build_step
    if not graph:
        for _ in range(2):
            step.refresh()
            expected = step.serial_tables()
            actual = step.forward(layers, tables, CUDAGraphMode.NONE)
            step.clear_tables()
            _assert_tables_equal(actual, expected)
        return

    step.forward(layers, tables, CUDAGraphMode.NONE)  # compile
    torch.accelerator.synchronize()
    step.clear_tables()
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        actual = step.forward(layers, tables, CUDAGraphMode.FULL)
    for _ in range(3):
        step.refresh()
        expected = step.serial_tables()
        cuda_graph.replay()
        _assert_tables_equal(actual, expected)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_sparse_index_side_builds_join_before_topk_overwrite(side_build_step, graph):
    """Layer 2 forks the key-A tables of groups 3, 0 and 1, which no layer
    reads before the next index source joins and overwrites the top-k
    indices: the tables still hold the indices they were built from."""
    step, layers, tables = side_build_step

    def run(mode):
        step.forward(layers, tables, mode, stop=3, join_at_end=False)
        # Layer 8, the next index source, joins and overwrites the indices.
        tables.join()
        step.topk.copy_(step.sources[8])
        # Only layer 2's fork can have built the other groups' tables by now.
        return [tables._tables[(id(group), KEY_A)] for group in step.groups]

    def expected_tables():
        # Layers 2-5 are the first key-A layers of groups 2, 3, 0 and 1.
        serial = step.serial_tables()
        return [serial[2 + (group - 2) % GROUPS] for group in range(GROUPS)]

    if graph:
        run(CUDAGraphMode.NONE)  # compile
        torch.accelerator.synchronize()
        step.clear_tables()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            actual = run(CUDAGraphMode.FULL)
        step.refresh()
        expected = expected_tables()
        cuda_graph.replay()
    else:
        expected = expected_tables()
        actual = run(CUDAGraphMode.NONE)
    _assert_tables_equal(actual, expected)


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )
    )


@pytest.mark.parametrize(
    "layer_id, wide",
    [(0, False), (0, True), (2, False)],
    ids=["swa-only", "swa-only-dspark-window", "compressed"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sparse_mla_decode_launches_the_stock_decode(
    monkeypatch, dtype, layer_id, wide
):
    """``SparseMlaDecode.run`` launches the sparse-MLA decode with the
    arguments the stock ``_forward`` passes for a pure-decode step (padded
    query heads, padded output rows, DSpark's wide window), plus the KV
    counters."""
    step = SideBuildStep(seed=1)
    layer = step.layers()[layer_id]
    group = step.by_layer[layer.swa_cache_layer.prefix]
    if wide:
        width = WINDOW + 16
        group.decode_swa_width = width
        group.decode_swa_indices = torch.randint(
            -1, 500, (TOKENS, width), dtype=torch.int32, device=DEVICE
        )
        group.flashinfer_decode_topk_lens = torch.full(
            (TOKENS,), width, dtype=torch.int32, device=DEVICE
        )
        group.flashinfer_decode_seq_lens = torch.full(
            (TOKENS,), width, dtype=torch.int32, device=DEVICE
        )
    heads, padded_heads, head_dim = 32, 64, 512
    layer.kv_cache_torch_dtype = dtype
    layer.scale = head_dim**-0.5
    layer._flashinfer_fp8_bmm1_scale = 0.5
    layer._flashinfer_fp8_bmm2_scale = 2.0
    layer.attn_sink = torch.zeros(padded_heads, device=DEVICE)
    layer._compressed_kv_cache = lambda: step.compressed_cache
    step.topk.copy_(step.sources[2])

    launches = []
    monkeypatch.setattr(
        flashinfer_sparse,
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
        lambda **kwargs: launches.append(kwargs),
    )
    monkeypatch.setattr(
        sparse_mla,
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
        lambda **kwargs: launches.append(kwargs),
    )
    q = torch.randn(TOKENS, heads, head_dim, device=DEVICE).to(dtype)
    # Padded rows past the step's tokens, as in a CUDA graph.
    out = torch.zeros(
        TOKENS + 2, padded_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    layer._forward(q, *step.lookup_args(layer), out)
    step.clear_tables()
    context = ForwardContext(
        {}, step.by_layer, {}, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    with override_forward_context(context):
        decode = SparseMlaDecode(SparseIndexTables([layer]), TOKENS * padded_heads)
        decode.run(layer, q, out)

    stock, executor = launches
    assert executor.pop("multi_ctas_kv_counter_buffer", None) is decode.kv_counters
    assert stock.keys() == executor.keys()
    for name, want in stock.items():
        got = executor[name]
        if isinstance(want, torch.Tensor):
            assert _same_bytes(got, want), name
        else:
            assert got == want, name


# ------------------------------------------------ metadata-build graphs


class _ChainBuilder:
    """A metadata builder that, like the DeepSeek-V4.1 builders, launches a
    short dependent chain into persistent buffers: the shared token-to-request
    map, an in-place block-table clamp, then gathers."""

    supports_update_block_table = False

    def __init__(self, offset: int, max_tokens: int, delay: int = 0):
        self.offset = offset
        # GPU cycles to wait before the clamp, so that a build that does not
        # wait for it reads the table first.
        self.delay = delay
        self.token_to_req = torch.zeros(max_tokens, dtype=torch.int32, device=DEVICE)
        self.out = torch.zeros(max_tokens, dtype=torch.int64, device=DEVICE)

    def build(self, common_prefix_len, common_attn_metadata):
        cm = common_attn_metadata
        token_to_req = cm.token_to_req_indices(self.token_to_req)
        if self.delay:
            torch.cuda._sleep(self.delay)
        block_table = cm.block_table_tensor.clamp_(min=0)
        req = token_to_req.long()
        value = block_table[:, 0].index_select(0, req) * 1000
        value = value + cm.seq_lens.index_select(0, req)
        for _ in range(4):
            value = value * 3 + cm.positions + self.offset
        out = self.out[: cm.num_actual_tokens]
        torch.add(value, cm.slot_mapping, out=out)
        return {"out": out, "token_to_req": token_to_req, "block_table": block_table}


class _ReaderBuilder(_ChainBuilder):
    """A builder that, like the indexer's, only reads its block table: it
    must see the table its group's writers clamped."""

    def build(self, common_prefix_len, common_attn_metadata):
        cm = common_attn_metadata
        token_to_req = cm.token_to_req_indices(self.token_to_req)
        req = token_to_req.long()
        value = cm.block_table_tensor[:, 0].index_select(0, req).long()
        for _ in range(4):
            value = value * 3 + cm.positions + self.offset
        out = self.out[: cm.num_actual_tokens]
        torch.add(value, cm.slot_mapping, out=out)
        return {"out": out, "token_to_req": token_to_req}


@pytest.mark.parametrize("readers", [False, True])
def test_metadata_graph_replay_matches_eager_build(monkeypatch, readers):
    """A recorded metadata build of five KV-cache groups (two sharing a block
    table, so the build forks into four branches) refreshes, on replay, every
    buffer the FULL forward graphs read to the eager build's values, also
    after an eager build of another batch left other values there. With
    readers, the two groups sharing a table each add a builder that only reads
    it, which builds on a branch of its own after the table's clamps."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime import (
        metadata_graphs,
    )
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.utils import AttentionGroup

    num_reqs, query_len, groups = 3, 4, 5
    tokens = num_reqs * query_len
    query_start = torch.arange(0, tokens + 1, query_len, dtype=torch.int32)
    # Another batch of the same token count, mapping the tokens differently.
    other_query_start = torch.tensor([0, 1, 2, tokens], dtype=torch.int32)
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    positions = torch.zeros(tokens, dtype=torch.int64, device=DEVICE)
    slot_mappings = torch.zeros(groups, tokens, dtype=torch.int64, device=DEVICE)
    tables = [
        torch.zeros(num_reqs, 8, dtype=torch.int32, device=DEVICE) for _ in range(4)
    ]
    block_tables = [tables[0], tables[1], tables[2], tables[3], tables[1]]
    builders = [
        _ChainBuilder(i, 64, delay=1_000_000 if readers else 0) for i in range(groups)
    ]
    attn_groups = [
        [
            AttentionGroup(
                backend=None,
                layer_names=[f"g{i}.l{j}" for j in range(2)],
                kv_cache_spec=None,
                kv_cache_group_id=i,
                metadata_builders=[builder],
            )
        ]
        for i, builder in enumerate(builders)
    ]
    if readers:
        monkeypatch.setattr(
            metadata_graphs, "_BLOCK_TABLE_READERS", frozenset({_ReaderBuilder})
        )
        for i in (1, 4):
            attn_groups[i].append(
                AttentionGroup(
                    backend=None,
                    layer_names=[f"g{i}.r{j}" for j in range(2)],
                    kv_cache_spec=None,
                    kv_cache_group_id=i,
                    metadata_builders=[_ReaderBuilder(10 + i, 64)],
                )
            )

    # The runner's persistent query-start buffer, which the recorded build reads.
    persistent_query_start = query_start.to(DEVICE)

    def build(groups_, query_start_cpu=query_start, query_start_gpu=None):
        if query_start_gpu is None:
            query_start_gpu = persistent_query_start
        return build_attn_metadata(
            attn_groups=groups_,
            num_reqs=num_reqs,
            num_tokens=tokens,
            query_start_loc_gpu=query_start_gpu,
            query_start_loc_cpu=query_start_cpu,
            max_query_len=query_len,
            seq_lens=seq_lens,
            max_seq_len=1024,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * groups),
            positions=positions,
        )

    def tensors(metadata):
        return {(name, k): v for name, m in metadata.items() for k, v in m.items()}

    graphs = AttnMetadataCudaGraphs(
        SimpleNamespace(use_cumem_cudagraph_pool=False), query_len
    )
    key = (num_reqs, num_reqs, tokens, tokens)
    graphs.capture(key, attn_groups, block_tables, build)
    # What the FULL forward graphs read: the buffers of the eager build.
    forward_view = tensors(build(attn_groups))
    for step in range(4):
        seq_lens.copy_(torch.tensor([5, 9, 2]) * (step + 1))
        positions.copy_(torch.arange(tokens) + step)
        slot_mappings.copy_(torch.randint(-1, 100, slot_mappings.shape))
        for table in tables:
            table.copy_(torch.randint(-1, 50, table.shape))
            table[0, 0] = -1  # The clamp changes what the readers see.
        inputs = [t.clone() for t in tables]
        # An eager step of another batch leaves its values in the buffers.
        build(attn_groups, other_query_start, other_query_start.to(DEVICE))
        replayed = {k: v.clone() for k, v in tensors(graphs.replay(key)).items()}
        seen = {k: v.clone() for k, v in forward_view.items()}
        for table, saved in zip(tables, inputs):
            table.copy_(saved)
        expected = tensors(build(attn_groups))
        assert expected.keys() == replayed.keys() == seen.keys()
        for k, want in expected.items():
            assert torch.equal(seen[k], want), (step, k)
            assert torch.equal(replayed[k], want), (step, k)


@pytest.mark.parametrize("rows", [1, 12, 48, 192])
@torch.inference_mode()
def test_stream_embedding_matches_the_lookup(rows):
    """The replicated table's lookup: each row of the embedding and of its
    four hc streams is bitwise the table's row of its token, and zeros for a
    token outside the table (as the sharded lookup's sum gives it); the
    outputs are written in full and nothing past them."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.embedding.replicated import (  # noqa: E501
        embed_streams,
    )

    torch.manual_seed(0)
    vocab, hidden, hc_mult = 1000, 5120, 4
    table = torch.randn(vocab, hidden, device="cuda").bfloat16()
    ids = torch.randint(0, vocab, (rows,), dtype=torch.int32, device="cuda")
    outside = torch.tensor([-1, vocab, vocab + 4096, vocab - 1], dtype=torch.int32)
    ids[: min(rows, 4)] = outside[: min(rows, 4)].cuda()
    want = torch.zeros(rows, hidden, device="cuda").bfloat16()
    inside = (ids >= 0) & (ids < vocab)
    want[inside] = table[ids[inside].long()]

    streams = torch.full((rows + 1, hc_mult, hidden), float("nan"), device="cuda")
    streams = streams.bfloat16()
    embedding = torch.full((rows + 1, hidden), float("nan"), device="cuda").bfloat16()
    embed_streams(ids, table, embedding[:rows], streams[:rows])
    for c in range(hc_mult):
        assert torch.equal(streams[:rows, c], want)
    assert streams[rows:].isnan().all()
    assert torch.equal(embedding[:rows], want)
    assert embedding[rows:].isnan().all()
