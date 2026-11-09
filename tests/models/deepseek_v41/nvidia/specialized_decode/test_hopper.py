# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper executor's sparse-MLA decode
(`specialized_decode/hopper/sparse_mla.py`) against the stock FlashMLA decode
it replaces, on the FlashMLA SM90 kernel, eagerly and replayed from a CUDA
graph. Needs one SM90 GPU; under a minute.

Run: pytest -v tests/models/deepseek_v41/nvidia/specialized_decode/test_hopper.py
"""

import itertools
from types import SimpleNamespace

import pytest
import ray
import torch

from tests.utils import init_test_distributed_environment, multi_process_parallel
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.nvidia import flashmla
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import sparse_mla
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(90)),
    reason="needs an SM90 GPU",
)

DEVICE = torch.device("cuda")
GROUPS = 2  # SWA KV-cache groups; layer i is in group i % GROUPS.
# (compress_ratio, kv source, index source) per layer: two runs of index
# sources per ratio, each run spanning both groups.
LAYOUT = (
    (0, None, None),
    (0, None, None),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 4),
    (2, 2, 4),
    (1, 6, 6),
    (1, 6, 6),
    (1, 6, 8),
    (1, 6, 8),
)
SEQ_LENS = (300, 2000)
ROWS = 6  # DSpark: 6 verify rows a request
TOKENS = ROWS * len(SEQ_LENS)
# Padded heads; a TP4 rank's 16 local heads are the first.
HEADS, LOCAL_HEADS, HEAD_DIM, WINDOW, TOP_K = 64, 16, 512, 128, 512
# Cache pages of 64 entries, as on SM90: a compressed page holds 64 * ratio
# tokens.
PAGE = 64


def _v4_cache(blocks: int, block_size: int, generator) -> torch.Tensor:
    """A V4 (584 B a token) FP8 cache of finite values: each block holds its
    data rows (448 e4m3 NoPE values, 64 BF16 RoPE values), then its scale rows
    (7 e8m0 scales and a pad byte)."""

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=DEVICE)

    nope = randn(blocks, block_size, 448).clamp(-6, 6).to(torch.float8_e4m3fn)
    rope = randn(blocks, block_size, 64).to(torch.bfloat16)
    scales = torch.randint(
        125, 130, (blocks, block_size, 8), generator=generator, device=DEVICE
    ).to(torch.uint8)
    scales[..., 7] = 0
    data = torch.cat([nope.view(torch.uint8), rope.view(torch.uint8)], dim=-1)
    block = torch.cat([data.flatten(1), scales.flatten(1)], dim=-1)
    return block.view(blocks, block_size, 584)


class Step:
    """One pure-decode step of `LAYOUT`'s layers on random caches."""

    def __init__(self, seed: int):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.generator = g
        positions = torch.cat(
            [torch.arange(n - ROWS, n, device=DEVICE) for n in SEQ_LENS]
        )
        swa_lens = (positions + 1).clamp(max=WINDOW).to(torch.int32)
        self.topk = torch.zeros(TOKENS, TOP_K, dtype=torch.int32, device=DEVICE)
        self.layers: list[DeepseekV4FlashMLAAttention] = []
        self.by_layer: dict = {}
        self.groups = []
        for _ in range(GROUPS):
            cache = _v4_cache(16, PAGE, g)
            # Each group's own SWA slots, -1 past the window.
            slots = torch.stack(
                [torch.randperm(16 * PAGE, generator=g, device=DEVICE)[:WINDOW]]
                * TOKENS
            ).to(torch.int32)
            columns = torch.arange(WINDOW, device=DEVICE)
            slots[columns >= swa_lens[:, None]] = -1
            self.groups.append(
                SimpleNamespace(
                    num_decodes=len(SEQ_LENS),
                    num_prefills=0,
                    num_decode_tokens=TOKENS,
                    max_decode_query_len=ROWS,
                    is_valid_token=torch.ones(TOKENS, dtype=torch.bool, device=DEVICE),
                    token_to_req_indices=torch.arange(
                        len(SEQ_LENS), dtype=torch.int32, device=DEVICE
                    ).repeat_interleave(ROWS),
                    decode_swa_indices=slots.unsqueeze(1),
                    decode_swa_lens=swa_lens,
                    kv_cache=cache,
                )
            )
        self.compressed = {}
        for kv_source in {kv for _, kv, _ in LAYOUT if kv is not None}:
            ratio = LAYOUT[kv_source][0]
            self.compressed[kv_source] = (
                _v4_cache(64, PAGE, g),
                SimpleNamespace(
                    block_size=PAGE * ratio,
                    block_table=torch.randperm(64, generator=g, device=DEVICE)
                    .view(2, 32)
                    .to(torch.int32),
                ),
            )
            self.by_layer[f"layers.{kv_source}"] = self.compressed[kv_source][1]
        for layer_id, (ratio, kv_source, index_source) in enumerate(LAYOUT):
            group = self.groups[layer_id % GROUPS]
            layer = object.__new__(DeepseekV4FlashMLAAttention)
            layer.compress_ratio = ratio
            layer.n_local_heads, layer.padded_heads = LOCAL_HEADS, HEADS
            layer.index_source_layer_id = index_source
            layer.compressed_cache_prefix = (
                None if kv_source is None else f"layers.{kv_source}"
            )
            layer.topk_indices_buffer = self.topk
            layer.scale = HEAD_DIM**-0.5
            layer.attn_sink = torch.randn(HEADS, generator=g, device=DEVICE)
            prefix = f"layers.{layer_id}.swa_cache"
            layer.swa_cache_layer = SimpleNamespace(
                prefix=prefix, kv_cache=group.kv_cache
            )
            if kv_source is not None:
                cache = self.compressed[kv_source][0]
                layer._compressed_kv_cache = lambda cache=cache: cache  # type: ignore[method-assign,misc]
            self.by_layer[prefix] = group
            self.layers.append(layer)
        self.q = torch.empty(
            len(LAYOUT), TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        self.sources = {i: torch.empty_like(self.topk) for _, _, i in LAYOUT if i}
        self.refresh()

    def refresh(self) -> None:
        """New queries and new top-k indices from every index source: like the
        indexer, the first min(512, compressed length) entries of each row are
        distinct valid indices."""
        g = self.generator
        self.q.copy_(torch.randn(self.q.shape, generator=g, device=DEVICE))
        for source, topk in self.sources.items():
            ratio = LAYOUT[source][0]
            topk.fill_(-1)
            for row in range(TOKENS):
                valid = SEQ_LENS[row // ROWS] // ratio
                count = min(TOP_K, valid)
                picks = torch.randperm(valid, generator=g, device=DEVICE)[:count]
                topk[row, :count] = picks.to(torch.int32)

    def fresh_plans(self) -> None:
        """What a metadata build gives each group: empty per-type plans."""
        for group in self.groups:
            group.tile_sched_swaonly = sparse_mla.get_mla_metadata()[0]
            group.tile_sched_c1a = sparse_mla.get_mla_metadata()[0]
            group.tile_sched_c2a = sparse_mla.get_mla_metadata()[0]

    def forward(self, decode: "sparse_mla.FlashMlaDecode | None") -> torch.Tensor:
        """Every layer's attention output in model order, on `decode` (None:
        the stock decode); each index source first writes its top-k."""
        self.fresh_plans()
        out = torch.zeros_like(self.q)
        context = ForwardContext(
            {}, self.by_layer, {}, cudagraph_runtime_mode=CUDAGraphMode.FULL
        )
        with override_forward_context(context):
            if decode is not None:
                decode.tables.reset()
            for layer_id, layer in enumerate(self.layers):
                if layer_id in self.sources:
                    self.topk.copy_(self.sources[layer_id])
                if decode is not None:
                    decode.run(layer, self.q[layer_id], out[layer_id])
                    continue
                kv_source = LAYOUT[layer_id][1]
                layer._forward_decode(
                    q=self.q[layer_id],
                    kv_cache=None
                    if kv_source is None
                    else layer._compressed_kv_cache(),
                    swa_metadata=self.by_layer[layer.swa_cache_layer.prefix],
                    attn_metadata=(
                        None if kv_source is None else self.compressed[kv_source][1]
                    ),
                    swa_only=kv_source is None,
                    output=out[layer_id],
                )
        return out


def _count_plans(monkeypatch, module) -> list:
    """The plans `module`'s decode launches pass, in launch order."""
    plans: list = []
    launch = module.flash_mla_with_kvcache

    def counted(**kwargs):
        plans.append(kwargs["tile_scheduler_metadata"])
        return launch(**kwargs)

    monkeypatch.setattr(module, "flash_mla_with_kvcache", counted)
    return plans


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_shared_plans_and_tables_match_the_stock_decode(monkeypatch, graph):
    """Every layer's output equals the stock decode's bit for bit, on one plan
    per layer type for both SWA groups (the stock decode plans each type once
    per group) and one global top-k table per index source; a CUDA graph of
    the step replays on new queries and top-k indices."""
    step = Step(seed=0)
    stock_plans = _count_plans(monkeypatch, flashmla)
    shared_plans = _count_plans(monkeypatch, sparse_mla)
    builds: list = []
    compute = sparse_mla.compute_global_topk_indices_and_lens

    def counted_compute(*args, **kwargs):
        builds.append(args)
        return compute(*args, **kwargs)

    monkeypatch.setattr(
        sparse_mla, "compute_global_topk_indices_and_lens", counted_compute
    )
    decode = sparse_mla.FlashMlaDecode(
        sparse_mla.FlashMlaTables(step.layers, small=False)
    )

    expected = step.forward(None)
    assert len({id(plan) for plan in stock_plans}) == 3 * GROUPS
    if not graph:
        actual = step.forward(decode)
    else:
        step.forward(decode)  # compile
        torch.accelerator.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            actual = step.forward(decode)
        step.refresh()
        expected = step.forward(None)
        cuda_graph.replay()
    assert len({id(plan) for plan in shared_plans[-len(LAYOUT) :]}) == 3
    assert len(builds) == (2 if graph else 1) * len(step.sources)
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected)


def _assert_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Each layer's output within one BF16 ulp of its largest magnitude, and
    within the relative error of one ulp: the kernels sum in another order
    than FlashMLA, so elements near zero differ by more of their own ulps."""
    for layer, (a, e) in enumerate(zip(actual.float(), expected.float())):
        largest = e.abs().max()
        ulp = torch.exp2(torch.floor(torch.log2(largest)) - 7)
        error = (a - e).abs().max()
        assert error <= ulp, (layer, error.item(), ulp.item())
        assert (a - e).norm() / e.norm() < 2**-8, layer


def _rope_cache(positions: int = 4096) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(7)
    angles = torch.rand(positions, 32, generator=g, device=DEVICE) * 6.28
    return torch.cat([angles.cos(), angles.sin()], -1).contiguous()


class ChainStep:
    """One uniform verify step of `requests` 6-row requests at `context`
    tokens on one compressed (ratio 2) layer, with SWA windows and slots as
    the stock metadata derives them (a per-request ring of 256 slots)."""

    def __init__(self, requests: int, context: int, seed: int = 0):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.rows = rows = requests * ROWS
        lens = [context - 37 * b for b in range(requests)]
        positions = torch.cat([torch.arange(n - ROWS, n, device=DEVICE) for n in lens])
        request = torch.arange(rows, device=DEVICE) // ROWS
        ring = 4 * PAGE
        self.positions = positions
        self.slots = (request * ring + positions % ring).long()
        swa = torch.full((rows, WINDOW), -1, dtype=torch.int32, device=DEVICE)
        columns = torch.arange(WINDOW, device=DEVICE)
        first = (positions - WINDOW + 1).clamp(min=0)
        window = first[:, None] + columns
        swa = torch.where(
            window <= positions[:, None],
            (request[:, None] * ring + window % ring).int(),
            swa,
        )
        self.swa_cache = _v4_cache(requests * 4, PAGE, g)
        pages = -(-context // 2 // PAGE)
        self.compressed_cache = _v4_cache(requests * pages, PAGE, g)
        table = torch.randperm(requests * pages, generator=g, device=DEVICE)
        self.metadata = SimpleNamespace(
            block_size=PAGE * 2,
            block_table=table.view(requests, pages).int(),
        )
        self.swa_metadata = SimpleNamespace(
            num_decodes=requests,
            num_prefills=0,
            num_decode_tokens=rows,
            max_decode_query_len=ROWS,
            is_valid_token=torch.ones(rows, dtype=torch.bool, device=DEVICE),
            token_to_req_indices=request.int(),
            decode_swa_indices=swa.unsqueeze(1),
            decode_swa_lens=(positions + 1).clamp(max=WINDOW).int(),
            slot_mapping=self.slots,
            block_size=PAGE,
        )
        # Each row's top-k: 70% of a per-request base set, the rest its own.
        topk = torch.empty(rows, TOP_K, dtype=torch.int32, device=DEVICE)
        for b, n in enumerate(lens):
            base = torch.randperm(n // 2, generator=g, device=DEVICE)[:TOP_K]
            for r in range(ROWS):
                own = torch.randperm(n // 2, generator=g, device=DEVICE)
                picks = torch.cat([base[: TOP_K * 7 // 10], own])
                picks = torch.unique(picks, sorted=False)[:TOP_K]
                order = torch.randperm(TOP_K, generator=g, device=DEVICE)
                topk[b * ROWS + r] = picks[order]
        cos_sin = _rope_cache(context + 1)
        layer = object.__new__(DeepseekV4FlashMLAAttention)
        layer.compress_ratio = 2
        layer.n_local_heads, layer.padded_heads = LOCAL_HEADS, HEADS
        layer.n_local_groups = 2
        layer.index_source_layer_id = 2
        layer.compressed_cache_prefix = "layers.2"
        layer.topk_indices_buffer = topk
        layer.scale = HEAD_DIM**-0.5
        layer.attn_sink = torch.randn(HEADS, generator=g, device=DEVICE)
        layer.eps = 1e-6
        layer.q_norm = SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(1280, generator=g, device=DEVICE)).bfloat16()
        )
        layer.kv_norm = SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(512, generator=g, device=DEVICE)).bfloat16()
        )
        layer.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)
        layer.swa_cache_layer = SimpleNamespace(
            prefix="layers.2.swa_cache", kv_cache=self.swa_cache
        )
        layer._compressed_kv_cache = lambda: self.compressed_cache  # type: ignore[method-assign]
        self.layer = layer
        self.by_layer = {
            "layers.2.swa_cache": self.swa_metadata,
            "layers.2": self.metadata,
        }
        self.qr_kv = (
            torch.randn(rows, 1792, generator=g, device=DEVICE) * 3
        ).bfloat16()
        self.q = torch.randn(rows, LOCAL_HEADS, HEAD_DIM, generator=g, device=DEVICE)
        self.q = self.q.bfloat16()

    def context(self) -> ForwardContext:
        return ForwardContext(
            {}, self.by_layer, {}, cudagraph_runtime_mode=CUDAGraphMode.FULL
        )


@pytest.mark.parametrize("context", [8192, 32768])
@pytest.mark.parametrize("requests", [1, 2, 4, 8])
def test_chain_decode_matches_flashmla(requests, context):
    """The `attn_chain` steps' MLA (6 to 48 rows): the q/kv norm writing the
    step's keys into the SWA tile images and the cluster decode on the tile
    images (query RoPE as it stages the `wq_b` output, splits merged in the
    cluster, inverse RoPE) against `fused_q_kv_rmsnorm`, the stock q RoPE
    and KV insert, FlashMLA and `fused_inv_rope_fp8_quant`: the normalized
    query and the whole SWA cache bit for bit, the `wo_a` input within one
    BF16 ulp of FlashMLA's; a CUDA graph replay equals the eager step bit for
    bit."""
    import vllm._custom_ops  # noqa: F401
    from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import mla_small

    step = ChainStep(requests, context)
    layer, rows = step.layer, step.rows
    cache0 = step.swa_cache.clone()
    small = mla_small.SmallMlaDecode([layer], DEVICE)
    small.warm_up()

    def rotate(out: torch.Tensor) -> torch.Tensor:
        return fused_inv_rope_fp8_quant(
            out[:, :LOCAL_HEADS],
            step.positions,
            layer.rotary_emb.cos_sin_cache,
            n_groups=2,
            heads_per_group=LOCAL_HEADS // 2,
            quantize=False,
        )[0]

    with override_forward_context(step.context()):
        # The stock ops.
        qr_in, kv_in = step.qr_kv.split([1280, 512], -1)
        qr, kv = fused_q_kv_rmsnorm(
            qr_in,
            kv_in,
            layer.q_norm.weight,
            layer.kv_norm.weight,
            layer.eps,
        )
        q = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            step.q,
            kv,
            step.swa_cache.view(step.swa_cache.shape[0], -1),
            step.slots,
            step.positions,
            layer.rotary_emb.cos_sin_cache,
            64,
            layer.eps,
            PAGE,
            False,
            False,
            True,
            False,
        )
        stock_cache = step.swa_cache.clone()
        decode = sparse_mla.FlashMlaDecode(sparse_mla.FlashMlaTables([layer], False))
        decode.tables.reset()
        flash = torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        decode.run(layer, q, flash)
        want = rotate(flash)

        def chain() -> tuple[torch.Tensor, torch.Tensor]:
            small.start()
            qr = small.chain_norm(layer, step.qr_kv, step.positions, step.swa_metadata)
            return qr, small.run_chain(
                layer, step.q, step.swa_metadata, step.metadata, step.positions
            )

        step.swa_cache.copy_(cache0)
        qr_chain, rotated = chain()
        torch.accelerator.synchronize()
        assert torch.equal(qr_chain, qr)
        assert torch.equal(step.swa_cache, stock_cache)
        rotated = rotated.clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qr_graph, rotated_graph = chain()
        step.swa_cache.copy_(cache0)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(qr_graph, qr)
        assert torch.equal(step.swa_cache, stock_cache)
        assert torch.equal(rotated_graph, rotated)
    torch.accelerator.synchronize()
    _assert_within_one_ulp(rotated[None], want[None])


# ------------------------------------------------------------- dense FP8 GEMMs

# Per seam: (N, K) per TP4 rank.
DENSE_SHAPES = {
    "wqa_wkv": (1792, 5120),
    "wq_b": (8192, 1280),
    "index_wq_b": (4096, 1280),
    "wo_b": (5120, 2048),
    "shared_gate_up": (1152, 5120),
    "shared_down": (5120, 576),
}


def _mxfp8_weight(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A random FP8 weight and UE8M0 scales per 32 k, alike over 32 rows as in
    the checkpoint."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEVICE).mul(100).clamp(-448, 448)
    scales = torch.randint(
        110, 122, (-(-n // 32), k // 32), generator=g, device=DEVICE
    ).to(torch.uint8)
    return w.to(torch.float8_e4m3fn), scales.repeat_interleave(32, 0)[:n].contiguous()


def _dequant(w: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales.float() - 127)
    return (w.float().unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)).flatten(-2)


def _humming_linear(
    w: torch.Tensor, scales: torch.Tensor, parts: list[int] | None = None
):
    """`w` (output `parts`) as the stock SM90 Humming W8A16 linear holds
    it: the weight and scales and the Humming kernel's config and locks."""
    from torch import nn

    from vllm.model_executor.layers.quantization.utils.humming import (
        convert_linear_layer_to_humming_standard,
        get_humming_linear_compute_config,
        prepare_humming_linear_layer_config,
        quant_key_to_input_schema,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )
    from vllm.utils.humming import create_humming_locks

    n, k = w.shape
    layer = nn.Module()
    layer.input_size_per_partition = layer.input_size = k
    layer.output_partition_sizes = parts or [n]
    layer.params_dtype = torch.bfloat16
    layer.has_bias = False
    layer.weight = nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(
        scales.clone().view(torch.float8_e8m0fnu), requires_grad=False
    )
    convert_linear_layer_to_humming_standard(
        layer=layer, name_map={"weight": "weight", "weight_scale": "weight_scale"}
    )
    config = prepare_humming_linear_layer_config(
        layer,
        {
            "quant_method": "humming",
            "dtype": "float8e4m3",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        },
        input_schema=quant_key_to_input_schema(kMxfp8Dynamic),
    )
    return layer, SimpleNamespace(
        layer_config=config,
        compute_config=get_humming_linear_compute_config(),
        locks=create_humming_locks(DEVICE),
    )


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("seam", list(DENSE_SHAPES))
def test_dense_calls_match_the_stock_linear(seam):
    """Every row count a seam serves: its call (CuTe DSL or tuned Humming, as
    dispatched) against an FP32 product of the exact dequantized weight, and
    against the stock Humming linear; strided rows as the executor passes
    them; the CuTe calls are deterministic."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        apply_humming_linear,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import dense

    n, k = DENSE_SHAPES[seam]
    w, scales = _mxfp8_weight(n, k, seed=n + k)
    layer, humming = _humming_linear(w, scales)
    linear = dense.DenseFp8Linear(seam, layer.weight, layer.weight_scale, n, k, humming)
    reference_weight = _dequant(w, scales)
    served = [m for m in range(1, dense.MAX_ROWS + 1) if linear.serves(m)]
    assert served == list(range(1, max(dense.DISPATCH[seam]) + 1))
    assert not linear.serves(max(dense.DISPATCH[seam]) + 1)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for m in served:
        x = torch.randn(m, k + 64, generator=g, device=DEVICE).bfloat16()[:, :k]
        want = x.float() @ reference_weight.t()
        got = linear(x)
        stock = apply_humming_linear(
            layer,
            x,
            skip_bias_add=True,
            layer_config=humming.layer_config,
            compute_config=humming.compute_config,
            locks=humming.locks,
        )
        assert _rel(got, want) < 5e-3, (m, _rel(got, want))
        assert _rel(got, stock) < 8e-3, (m, _rel(got, stock))
        if isinstance(dense.choice(seam, m), dense.Cute):
            assert torch.equal(linear(x), got), m


def test_fp8_wo_a_matches_the_stock_bmm():
    """The FP8 copy of the stock BF16 `wo_a` (dequantized at load) is exact,
    and its grouped call on the inverse RoPE's group-major output matches the
    stock BF16 bmm at every row count; a weight that is not an MXFP8 dequant
    keeps the bmm."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import dense

    groups, rank, k = 2, 1024, 4096
    w, scales = _mxfp8_weight(groups * rank, k, seed=1)
    bf16 = _dequant(w, scales).bfloat16()
    linear = dense.fp8_wo_a(SimpleNamespace(weight=bf16), groups)
    assert linear is not None
    not_mxfp8 = SimpleNamespace(weight=bf16 + 2.0**-7 * bf16.abs().max())
    assert dense.fp8_wo_a(not_mxfp8, groups) is None
    grouped = bf16.view(groups, rank, k)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    served = max(dense.DISPATCH["wo_a"])
    assert not linear.serves(served + 1)
    for m in range(1, served + 1):
        assert linear.serves(m)
        # The inverse RoPE writes [groups, tokens, k] and returns its transpose.
        rotated = torch.randn(groups, m, k, generator=g, device=DEVICE).bfloat16()
        x = rotated.transpose(0, 1)
        want = torch.einsum("tgk,gnk->tgn", x.float(), grouped.float()).flatten(1)
        stock = torch.bmm(rotated, grouped.transpose(1, 2)).transpose(0, 1)
        got = linear(x)
        assert _rel(got, want) < 5e-3, (m, _rel(got, want))
        assert _rel(got, stock.flatten(1)) < 8e-3, m


# ---------------------------------------------------------------- low-batch MoE

MOE_HIDDEN, MOE_INTER = 5120, 576
# (experts, top-k): the backbone's layers, and a smaller layer (the kernel
# takes the layer's expert count and top-k).
MOE_SHAPES = ((384, 6), (128, 3))
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_dequant(w: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), device=DEVICE)
    values = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1)
    scale = torch.exp2(scales.float() - 127).unsqueeze(-1)
    return (values.flatten(-2).unflatten(-1, (-1, 32)) * scale).flatten(-2)


def _stock_moe(num_experts: int, top_k: int):
    """A stock Humming W4A8 MoE layer (MXFP4 experts, block-FP8 activations)
    of one TP4 rank's shapes, on random checkpoint-format weights."""
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
    from vllm.model_executor.layers.quantization.mxfp4 import (
        Mxfp4Config,
        Mxfp4MoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    class Config(Mxfp4Config):
        def _make_moe_method(self, moe):
            return Mxfp4MoEMethod(moe, activation_key=kMxfp8Dynamic)

    oracle._user_moe_activation_override = lambda: None
    g = torch.Generator(device=DEVICE).manual_seed(0)

    def randint(low, high, *shape):
        return torch.randint(
            low, high, shape, generator=g, device=DEVICE, dtype=torch.uint8
        )

    shapes = (num_experts, 2 * MOE_INTER)
    w13 = randint(0, 256, *shapes, MOE_HIDDEN // 2)
    s13 = randint(120, 123, *shapes, MOE_HIDDEN // 32)
    w2 = randint(0, 256, num_experts, MOE_HIDDEN, MOE_INTER // 2)
    s2 = randint(121, 124, num_experts, MOE_HIDDEN, MOE_INTER // 32)
    with torch.device(DEVICE):
        layer = FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=MOE_HIDDEN,
            intermediate_size=MOE_INTER,
            params_dtype=torch.bfloat16,
            renormalize=True,
            quant_config=Config(),
            tp_size=1,
            dp_size=1,
            prefix="layers.0.experts",
            swiglu_limit=10.0,
            activation="silu",
        )
    experts = layer.routed_experts
    with torch.no_grad():
        for name, value in (("w13_weight", w13), ("w13_weight_scale", s13)):
            p = getattr(experts, name)
            p.zero_()
            half = p.shape[1] // 2
            p[:, :MOE_INTER, : value.shape[2]].copy_(value[:, :MOE_INTER])
            p[:, half : half + MOE_INTER, : value.shape[2]].copy_(value[:, MOE_INTER:])
        for name, value in (("w2_weight", w2), ("w2_weight_scale", s2)):
            p = getattr(experts, name)
            p.zero_()
            p[:, : value.shape[1], : value.shape[2]].copy_(value)
    layer._quant_method.process_weights_after_loading(experts)
    reference = (_mxfp4_dequant(w13, s13), _mxfp4_dequant(w2, s2))
    return layer, reference


def _routing(logits: torch.Tensor, bias: torch.Tensor, top_k: int):
    """DeepSeek-V4 routing: sqrt-softplus scores, top-k of score + bias,
    renormalized, times 1.5."""
    scores = torch.nn.functional.softplus(logits).sqrt()
    ids = (scores + bias).topk(top_k, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids.int(), (weights / weights.sum(-1, keepdim=True) * 1.5).float()


def _finalize(routed) -> torch.Tensor:
    """`AllReduceMHC.finalize`'s top-k sum on one rank."""
    rows = routed.gemm2_permuted[routed.expanded_idx_to_permuted_idx.long()]
    return (rows.float() * routed.expert_weights.unsqueeze(-1)).sum(1)


def _moe_ffn(num_experts: int, top_k: int, bias: torch.Tensor):
    """A `DeepseekV4MoE` stand-in on a stock Humming W4A8 MoE and a W8A16
    shared expert, with the routed experts' FP32 reference weights and the
    shared expert's Humming linears."""
    layer, reference = _stock_moe(num_experts, top_k)
    gate_up_w, gate_up_s = _mxfp8_weight(2 * MOE_INTER, MOE_HIDDEN, seed=3)
    down_w, down_s = _mxfp8_weight(MOE_HIDDEN, MOE_INTER, seed=4)
    gate_up = _humming_linear(gate_up_w, gate_up_s, [MOE_INTER, MOE_INTER])
    down = _humming_linear(down_w, down_s)
    ffn = SimpleNamespace(
        experts=layer,
        n_activated_experts=top_k,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        use_native_mega_moe=False,
        image_sentinel_lo=0,
        shared_experts=SimpleNamespace(gate_up_proj=gate_up[0], down_proj=down[0]),
        gate=SimpleNamespace(e_score_correction_bias=bias, bias_vl=None, tid2eid=None),
    )
    return ffn, reference, (gate_up, down)


def _moe_reference(x, ids, weights, w13, w2) -> torch.Tensor:
    """The routed experts' FP32 output, intermediate rounded to BF16."""
    want = torch.zeros(x.shape[0], MOE_HIDDEN, device=DEVICE)
    for k in range(ids.shape[1]):
        for e in ids[:, k].unique().tolist():
            rows = (ids[:, k] == e).nonzero().squeeze(1)
            h = x[rows].float() @ w13[e].t()
            a = torch.nn.functional.silu(h[:, :MOE_INTER].clamp(max=10.0))
            a = (a * h[:, MOE_INTER:].clamp(-10.0, 10.0)).bfloat16().float()
            want[rows] += (a @ w2[e].t()) * weights[rows, k, None]
    return want


@pytest.mark.parametrize(
    "num_experts,top_k", MOE_SHAPES, ids=[f"e{e}k{k}" for e, k in MOE_SHAPES]
)
def test_low_batch_moe_matches_the_stock_moe(
    dist_init, workspace_init, num_experts, top_k
):
    """The low-batch MoE kernel's unfinalized output, reduced as the
    boundary's finalize reduces it, against the stock Humming W4A8 MoE on the
    same routing at the row counts the executor serves, dense and entry mode,
    routed from logits and from given top-k; its in-kernel routing against
    the DeepSeek-V4 routing; and its fused shared expert against the stock
    W8A16 shared expert MLP; for the backbone's shape and a smaller one."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.quantization.utils.humming import (
        apply_humming_linear,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import moe

    config = VllmConfig(kernel_config=KernelConfig(moe_backend="humming"))
    with set_current_vllm_config(config):
        g = torch.Generator(device=DEVICE).manual_seed(1)
        bias = torch.randn(num_experts, generator=g, device=DEVICE) * 0.05
        ffn, (w13, w2), (gate_up, down) = _moe_ffn(num_experts, top_k, bias)
        layer = ffn.experts
        by_logits = moe.HopperLowBatchMoE.of(ffn)
        # Hash-routed: the kernel takes given top-k ids.
        ffn.gate.tid2eid = torch.zeros(1)
        by_ids = moe.HopperLowBatchMoE.of(ffn)
        assert by_logits is not None and by_ids is not None
        act = SiluAndMulWithClamp(10.0)
        assert not by_logits.serves(moe.MAX_ROWS + 1)
        for m in (1, 2, 6, 8, 9, 12, 16, 17, 24, 48, 96, 120, 144):
            assert by_logits.serves(m)
            x = torch.randn(m, MOE_HIDDEN, generator=g, device=DEVICE).bfloat16()
            logits = torch.randn(m, num_experts, generator=g, device=DEVICE)
            ids, weights = _routing(logits, bias, top_k)
            with set_forward_context(None, config, num_tokens=m):
                stock = layer._quant_method.apply(
                    layer.routed_experts, x, weights, ids, None, None
                ).float()
            want = _moe_reference(x, ids, weights, w13, w2)
            stock_error = _rel(stock, want)
            assert stock_error < 6e-2, m

            routed, shared = by_logits(x, router_logits=logits)
            got = _finalize(routed)
            kernel_ids = by_logits.workspace.ids[: m * top_k].view(m, top_k)
            assert torch.equal(kernel_ids.sort(-1).values, ids.sort(-1).values), m
            assert torch.allclose(
                routed.expert_weights.sort(-1).values,
                weights.sort(-1).values,
                atol=1e-5,
            ), m
            assert _rel(got, want) < 1.25 * stock_error, m

            routed, _ = by_ids(x, topk_ids=ids, topk_weights=weights)
            assert _rel(_finalize(routed), want) < 1.25 * stock_error, m

            if m <= moe.SHARED_MAX_ROWS:
                assert shared is not None
                stock_shared = x
                for (linear, humming), f in ((gate_up, act), (down, None)):
                    stock_shared = apply_humming_linear(
                        linear,
                        stock_shared,
                        skip_bias_add=True,
                        layer_config=humming.layer_config,
                        compute_config=humming.compute_config,
                        locks=humming.locks,
                    )
                    stock_shared = f(stock_shared) if f else stock_shared
                assert _rel(shared, stock_shared) < 1e-2, (
                    m,
                    _rel(shared, stock_shared),
                )
            else:
                assert shared is None


def test_low_batch_moe_every_row_count_skewed_and_spread(dist_init, workspace_init):
    """The low-batch MoE (dense, cluster and entry paths, routed
    in-kernel and by given ids) at every row count it serves, on the two
    extremes of routing: skewed (one expert takes every row, so its row chunks
    fill up, the others get a row or two) and spread (every route its own
    expert while there are experts left: the most tiles, so the most split-K
    partial slots). Each call against the FP32 reference, synchronized so an
    illegal access surfaces at its call."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import moe

    num_experts, top_k = MOE_SHAPES[0]
    config = VllmConfig(kernel_config=KernelConfig(moe_backend="humming"))
    with set_current_vllm_config(config):
        bias = torch.zeros(num_experts, device=DEVICE)
        ffn, (w13, w2), _ = _moe_ffn(num_experts, top_k, bias)
        mod = moe.HopperLowBatchMoE.of(ffn)
        assert mod is not None
        g = torch.Generator(device=DEVICE).manual_seed(11)
        for m, spread in itertools.product(range(1, moe.MAX_ROWS + 1), (False, True)):
            x = torch.randn(m, MOE_HIDDEN, generator=g, device=DEVICE).bfloat16()
            logits = torch.randn(m, num_experts, generator=g, device=DEVICE) - 4.0
            if spread:
                route = torch.arange(m * top_k, device=DEVICE).view(m, top_k)
                logits.scatter_(1, route % num_experts, 8.0)
            else:
                # expert 7 first for every row, then 5 of a small pool
                logits[:, 7] = 8.0
                pool = torch.randint(0, 24, (m, top_k - 1), generator=g, device=DEVICE)
                logits.scatter_(1, pool, 4.0)
            ids, weights = _routing(logits, bias, top_k)
            want = _moe_reference(x, ids, weights, w13, w2)
            for kw in (
                dict(router_logits=logits),
                dict(topk_ids=ids, topk_weights=weights),
            ):
                routed, _ = mod(x, **kw)
                got = _finalize(routed)
                torch.accelerator.synchronize()
                assert _rel(got, want) < 6e-2, (m, spread, list(kw), _rel(got, want))


def test_low_batch_moe_is_repeatable_on_a_dirty_workspace(dist_init, workspace_init):
    """Serving calls run on a workspace a different call left behind (another
    layer, row count, routing or kernel): layers of both shapes share
    it, and each target call at a serving shape (ids and logits routing, so
    the dense / cluster / entry paths) gives
    bitwise the same per-route rows, routing weights and shared-expert output
    after random sequences of other calls, eagerly and replayed from a CUDA
    graph, as on its first run."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import moe

    g = torch.Generator(device=DEVICE).manual_seed(13)
    layers = {}
    for num_experts, top_k in MOE_SHAPES:
        config = VllmConfig(kernel_config=KernelConfig(moe_backend="humming"))
        with set_current_vllm_config(config):
            bias = torch.randn(num_experts, generator=g, device=DEVICE) * 0.05
            mod = moe.HopperLowBatchMoE.of(_moe_ffn(num_experts, top_k, bias)[0])
        assert mod is not None
        layers[top_k] = (mod, num_experts, bias)
    assert layers[6][0].workspace is layers[3][0].workspace

    def make(top_k, m, spread, ids_mode):
        _, num_experts, bias = layers[top_k]
        x = torch.randn(m, MOE_HIDDEN, generator=g, device=DEVICE).bfloat16()
        logits = torch.randn(m, num_experts, generator=g, device=DEVICE)
        if spread:
            route = torch.arange(m * top_k, device=DEVICE).view(m, top_k)
            logits.scatter_(1, route % num_experts, 8.0)
        else:  # requests of 6 rows routed alike
            req = torch.randn((m + 5) // 6, num_experts, generator=g, device=DEVICE)
            logits = req.repeat_interleave(6, 0)[:m] + 0.5 * logits
        ids, weights = _routing(logits, bias, top_k)
        kw = (
            dict(topk_ids=ids, topk_weights=weights)
            if ids_mode
            else dict(router_logits=logits)
        )
        return top_k, x, kw

    def run(top_k, x, kw):
        routed, shared = layers[top_k][0](x, **kw)
        assert shared is not None
        rows = routed.gemm2_permuted[routed.expanded_idx_to_permuted_idx.long()]
        return rows.clone(), routed.expert_weights.clone(), shared.clone()

    shapes = [(6, m) for m in (6, 12, 24, 48, 144)] + [(3, m) for m in (5, 10, 40)]
    targets = [
        make(top_k, m, spread, ids_mode)
        for (top_k, m), spread, ids_mode in itertools.product(
            shapes, (False, True), (False, True)
        )
    ]
    others = [
        make(top_k, m, spread, ids_mode)
        for top_k, m, spread, ids_mode in itertools.product(
            (6, 3), (1, 4, 9, 16, 17, 30, 96, 140), (False, True), (False, True)
        )
    ]
    want = [run(*t) for t in targets]

    def same(got, ref):
        return all(torch.equal(a, b) for a, b in zip(got, ref))

    step = 0
    for i, target in enumerate(targets * 2):
        for _ in range(1 + step % 3):
            run(*others[(7 * step) % len(others)])
            step += 1
        assert same(run(*target), want[i % len(targets)]), i % len(targets)

    order = [(7 * j) % len(targets) for j in range(len(targets))]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = []
        for j, i in enumerate(order):
            run(*others[(5 * j) % len(others)])
            got.append(run(*targets[i]))
    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize()
        for i, out in zip(order, got):
            assert same(out, want[i]), i


@ray.remote(num_gpus=1, max_calls=1)
def _moe_feeds_the_fused_finalize(monkeypatch, tp_size, pp_size, rank, port):
    from vllm.config import (
        KernelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.all_reduce import (  # noqa: E501
        BoundaryAllReduce,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import moe
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.decode import (
        FUSED_MAX_ROWS,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    # Ray leaves CUDA_VISIBLE_DEVICES as the test's (multi_process_parallel), so
    # rank r runs on the r-th visible GPU.
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    config = VllmConfig(kernel_config=KernelConfig(moe_backend="humming"))
    with set_current_vllm_config(config):
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        num_experts, top_k = MOE_SHAPES[0]
        g = torch.Generator(device=device).manual_seed(17)
        bias = torch.randn(num_experts, generator=g, device=device) * 0.05
        with torch.accelerator.device_index(device.index):
            mod = moe.HopperLowBatchMoE.of(_moe_ffn(num_experts, top_k, bias)[0])
        assert mod is not None
        all_reduce = BoundaryAllReduce(
            SimpleNamespace(hidden_size=MOE_HIDDEN, hc_mult=4, num_experts_per_tok=6),
            quantize_layer_input=False,
            max_num_tokens=FUSED_MAX_ROWS,
            finalize_metadata_before_pdl=False,
        )
        assert all_reduce.bt_min_finalize_rows <= 96
        # The residual and mixes are replicated; each rank's MoE input
        # and routing its own.
        mixes_g = torch.Generator(device=device).manual_seed(5)
        g.manual_seed(100 + rank)

        def inputs(m):
            x = torch.randn(m, MOE_HIDDEN, generator=g, device=device).bfloat16()
            logits = torch.randn(m, num_experts, generator=g, device=device)
            mixes = (
                torch.randn(
                    m, 4, MOE_HIDDEN, generator=mixes_g, device=device
                ).bfloat16(),
                torch.rand(m, 4, generator=mixes_g, device=device),
                torch.randn(m, 4, 4, generator=mixes_g, device=device) * 0.1,
                torch.rand(m, 4, generator=mixes_g, device=device),
                torch.randn(MOE_HIDDEN, generator=mixes_g, device=device).bfloat16(),
            )
            return x, logits, mixes

        def boundary(x, logits, mixes):
            residual, post, comb, pre, weight = mixes
            routed, shared = mod(x, logits)
            out = all_reduce.finalize(
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
                shared,
                residual,
                post,
                comb,
                pre,
                weight,
                1e-6,
            )
            return routed, shared, out

        def reference(routed, shared, mixes):
            # One FP32 FMA per route in route order, the shared add, one
            # rounding (as the finalize sums), then the plain fused path.
            rows = routed.expanded_idx_to_permuted_idx
            acc = torch.zeros(rows.shape[0], MOE_HIDDEN, device=device)
            for k in range(top_k):
                valid = (rows[:, k] >= 0).unsqueeze(-1)
                routed_k = routed.gemm2_permuted[rows[:, k].clamp_min(0).long()]
                acc.addcmul_(
                    torch.where(valid, routed_k.float(), 0.0),
                    torch.where(valid, routed.expert_weights[:, k : k + 1], 0.0),
                )
            residual, post, comb, pre, weight = mixes
            reduced = (acc + shared.float()).bfloat16()
            return all_reduce(reduced, residual, post, comb, pre, weight, 1e-6)

        def same(got, want):
            return all(torch.equal(a, b) for a, b in zip(got, want))

        for m in (1, 6, 12, 48, 96, 144):
            calls = [inputs(m), inputs(m)]
            want = []
            for call in calls:
                routed, shared, out = boundary(*call)
                out = tuple(t.clone() for t in out)
                assert same(out, reference(routed, shared, call[2])), m
                want.append(out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = [boundary(*call)[2] for call in calls]
            for _ in range(3):
                graph.replay()
                torch.accelerator.synchronize()
                assert all(same(o, w) for o, w in zip(got, want)), m


def test_low_batch_moe_feeds_the_fused_finalize(monkeypatch):
    """The low-batch MoE kernel and the boundary's fused all-reduce + mHC
    finalize the executor launches right after it (PDL), at TP2: the finalize
    reads the route map and routing weights the MoE kernel writes only after
    its grid-dependency wait (`finalize_metadata_before_pdl=False`). Each
    call follows one of another routing, so a read of the previous call's map
    or weights shows; eagerly and replayed from a CUDA graph, LL and BT
    finalize, bit for bit against the plain fused all-reduce of the reduced
    output."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs two SM90 GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 2, 1, _moe_feeds_the_fused_finalize)


# ------------------------------------------------------------------ mHC pre

MHC_ROWS = (1, 6, 12, 24, 48, 96, 144, 192)


def _mhc_layer(seed: int) -> SimpleNamespace:
    """The decoder-layer attributes the mHC pre reads: both seams' FP32 `fn`
    / scale / base and norms, random."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=DEVICE) * scale

    def norm():
        weight = (1 + randn(MOE_HIDDEN, scale=0.1)).bfloat16()
        return SimpleNamespace(weight=weight, variance_epsilon=1e-6)

    layer = SimpleNamespace(
        attn_norm=norm(),
        ffn_norm=norm(),
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=20,
        hc_mult=4,
    )
    for seam in ("attn", "ffn"):
        setattr(layer, f"hc_{seam}_fn", randn(24, 4 * MOE_HIDDEN, scale=0.01))
        setattr(layer, f"hc_{seam}_scale", 0.5 + randn(3, scale=0.1).abs())
        setattr(layer, f"hc_{seam}_base", randn(24, scale=0.1))
    return layer


@pytest.mark.parametrize("first", [False, True], ids=["carried", "first"])
def test_mhc_pre_matches_torch(first):
    """The executor's mHC pre (the CuTe DSL mixes in place of the DeepGEMM
    prenorm GEMM, the stock epilogue) against `mhc_pre_delayed_torch` at
    every step size the executor serves, in chunks of 64 rows: the
    coefficients on the side stream and the normalized layer input on the
    caller's, with the carried pre-mix or (`first`) stream zero."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import mhc

    layer = _mhc_layer(seed=3)
    assert mhc.SeamPre.applies(layer)
    seam = mhc.SeamPre(layer, ffn=False)
    seam.warm_up(list(MHC_ROWS), first=True)
    side = torch.cuda.Stream()
    g = torch.Generator(device=DEVICE).manual_seed(5)
    for m in MHC_ROWS:
        residual = (
            torch.randn(m, 4, MOE_HIDDEN, generator=g, device=DEVICE) * 2
        ).bfloat16()
        pre = None if first else torch.rand(m, 4, generator=g, device=DEVICE)
        post, comb, x, next_pre = seam(residual, pre, side)
        torch.cuda.current_stream().wait_stream(side)
        want = mhc_pre_delayed_torch(
            residual,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=pre,
        )
        collapsed = want[2].float()
        want_x = (
            collapsed
            * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
            * layer.attn_norm.weight.float()
        )
        # TF32 products, as the DeepGEMM prenorm GEMM's: within 5e-4 of FP32.
        for got, ref in zip((post, comb, next_pre), (want[0], want[1], want[3])):
            assert _rel(got, ref) < 5e-4, (m, _rel(got, ref))
        assert _rel(x, want_x) < 1e-2, (m, _rel(x, want_x))
