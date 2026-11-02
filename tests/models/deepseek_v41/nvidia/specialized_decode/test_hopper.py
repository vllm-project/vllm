# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper executor's sparse-MLA decode
(``specialized_decode/hopper/sparse_mla.py``) against the stock FlashMLA decode
it replaces, on the FlashMLA SM90 kernel, eagerly and replayed from a CUDA
graph. Needs one SM90 GPU; under a minute.

Run: pytest -v tests/models/deepseek_v41/nvidia/specialized_decode/test_hopper.py
"""

from types import SimpleNamespace

import pytest
import torch

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
    """One pure-decode step of ``LAYOUT``'s layers on random caches."""

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
                layer._compressed_kv_cache = lambda cache=cache: cache
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
        """Every layer's attention output in model order, on ``decode`` (None:
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
    """The plans ``module``'s decode launches pass, in launch order."""
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


def _check_rotated_combine(step: Step, decode, out: torch.Tensor) -> None:
    """The small path's combine with the inverse RoPE folded in equals its
    plain combine followed by the stock ``fused_inv_rope_fp8_quant``."""
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    cos_sin = _rope_cache()
    positions = torch.arange(TOKENS, device=DEVICE) * 37 % 4096
    context = ForwardContext(
        {}, step.by_layer, {}, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    with override_forward_context(context):
        decode.tables.reset()
        for layer_id, layer in enumerate(step.layers):
            layer.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)
            layer.n_local_groups = 2
            if layer_id in step.sources:
                step.topk.copy_(step.sources[layer_id])
            rotated = decode.run_rotated(layer, step.q[layer_id], positions)
            want, _ = fused_inv_rope_fp8_quant(
                out[layer_id][:, :LOCAL_HEADS],
                positions,
                cos_sin,
                n_groups=2,
                heads_per_group=LOCAL_HEADS // 2,
                quantize=False,
            )
            assert torch.equal(rotated, want), layer_id


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_small_batch_decode_matches_the_stock_decode(monkeypatch, graph):
    """Two 6-row verify requests take the small-batch kernels: no FlashMLA
    launch and no global top-k table; every layer's local heads are within one
    BF16 ulp of the stock decode's at the layer's output scale, and a CUDA
    graph of the step replays on new queries and top-k indices."""
    step = Step(seed=0)
    launches = _count_plans(monkeypatch, sparse_mla)
    builds: list = []
    monkeypatch.setattr(
        sparse_mla,
        "compute_global_topk_indices_and_lens",
        lambda *args, **kwargs: builds.append(args),
    )
    decode = sparse_mla.FlashMlaDecode(sparse_mla.FlashMlaTables(step.layers))
    assert decode.tables.small is not None
    decode.tables.small.warm_up()

    expected = step.forward(None)
    if not graph:
        actual = step.forward(decode)
    else:
        step.forward(decode)
        torch.accelerator.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            actual = step.forward(decode)
        step.refresh()
        expected = step.forward(None)
        cuda_graph.replay()
    torch.accelerator.synchronize()
    assert not launches and not builds
    if not graph:
        _check_rotated_combine(step, decode, actual)
    local = slice(0, LOCAL_HEADS)
    assert torch.isfinite(expected).all()
    _assert_within_one_ulp(actual[:, :, local], expected[:, :, local])


def _rope_cache(positions: int = 4096) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(7)
    angles = torch.rand(positions, 32, generator=g, device=DEVICE) * 6.28
    return torch.cat([angles.cos(), angles.sin()], -1).contiguous()


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_attention_chain_kernels_match_the_stock_ops(rows):
    """The early-trigger q/kv RMSNorm with the KV insert and the query RoPE
    equal the stock ``fused_q_kv_rmsnorm`` and
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` bit for bit: the
    normalized query, the inserted V4 records and the rotated live heads."""
    import vllm._custom_ops  # noqa: F401
    from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import attn_chain

    g = torch.Generator(device=DEVICE).manual_seed(rows)
    eps, block_size, blocks = 1e-6, PAGE, 16
    qr_kv = (torch.randn(rows, 1792, generator=g, device=DEVICE) * 3).bfloat16()
    q_weight = (1 + 0.1 * torch.randn(1280, generator=g, device=DEVICE)).bfloat16()
    kv_weight = (1 + 0.1 * torch.randn(512, generator=g, device=DEVICE)).bfloat16()
    cos_sin = _rope_cache()
    positions = torch.randint(0, 4096, (rows,), generator=g, device=DEVICE)
    slots = torch.randperm(blocks * block_size, generator=g, device=DEVICE)[:rows]
    stock_cache = _v4_cache(blocks, block_size, g)
    chain_cache = stock_cache.clone()
    attn = SimpleNamespace(
        eps=eps,
        q_norm=SimpleNamespace(weight=q_weight),
        kv_norm=SimpleNamespace(weight=kv_weight),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        swa_cache_layer=SimpleNamespace(kv_cache=chain_cache),
    )
    metadata = SimpleNamespace(slot_mapping=slots, block_size=block_size)
    q = torch.randn(rows, 16, 512, generator=g, device=DEVICE).bfloat16()

    qr, kv = fused_q_kv_rmsnorm(*qr_kv.split([1280, 512], -1), q_weight, kv_weight, eps)
    q_stock = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        stock_cache.view(blocks, -1),
        slots,
        positions,
        cos_sin,
        64,
        eps,
        block_size,
        False,
        False,
        True,
        False,
    )
    qr_chain = attn_chain.norm_kv_insert(attn, qr_kv, positions, metadata)
    q_chain = attn_chain.q_rope(attn, q, positions)
    torch.accelerator.synchronize()
    assert torch.equal(qr_chain, qr)
    assert torch.equal(chain_cache, stock_cache)
    assert torch.equal(q_chain[:, :16], q_stock[:, :16])


# ------------------------------------------------------------- dense FP8 GEMMs

# Per seam: (N, K) per TP4 rank.
DENSE_SHAPES = {
    "wqa_wkv": (1792, 5120),
    "wq_b": (8192, 1280),
    "index_wq_b": (4096, 1280),
    "wo_b": (5120, 2048),
    "shared_gate_up": (1152, 5120),
    "shared_down": (5120, 576),
    "engram_wkv": (6400, 6144),
    "main_proj": (5120, 15360),
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
    """``w`` (output ``parts``) as the stock SM90 Humming W8A16 linear holds
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
        if linear.publishes(m):
            assert torch.equal(linear(x), got), m


def test_fp8_wo_a_matches_the_stock_bmm():
    """The FP8 copy of the stock BF16 ``wo_a`` (dequantized at load) is exact,
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
# (experts, top-k): the backbone's layers and the DSpark drafter's.
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
    """``AllReduceMHC.finalize``'s top-k sum on one rank."""
    rows = routed.gemm2_permuted[routed.expanded_idx_to_permuted_idx.long()]
    return (rows.float() * routed.expert_weights.unsqueeze(-1)).sum(1)


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
    W8A16 shared expert MLP; for the backbone's and the drafter's shapes."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.quantization.utils.humming import (
        apply_humming_linear,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import moe
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.draft import (
        finalize,
    )

    config = VllmConfig(kernel_config=KernelConfig(moe_backend="humming"))
    with set_current_vllm_config(config):
        layer, (w13, w2) = _stock_moe(num_experts, top_k)
        gate_up_w, gate_up_s = _mxfp8_weight(2 * MOE_INTER, MOE_HIDDEN, seed=3)
        down_w, down_s = _mxfp8_weight(MOE_HIDDEN, MOE_INTER, seed=4)
        gate_up, gate_up_humming = _humming_linear(
            gate_up_w, gate_up_s, [MOE_INTER, MOE_INTER]
        )
        down, down_humming = _humming_linear(down_w, down_s)
        g = torch.Generator(device=DEVICE).manual_seed(1)
        bias = torch.randn(num_experts, generator=g, device=DEVICE) * 0.05

        def ffn(hash_routed: bool):
            return SimpleNamespace(
                experts=layer,
                n_activated_experts=top_k,
                scoring_func="sqrtsoftplus",
                renormalize=True,
                routed_scaling_factor=1.5,
                swiglu_limit=10.0,
                use_native_mega_moe=False,
                image_sentinel_lo=0,
                shared_experts=SimpleNamespace(gate_up_proj=gate_up, down_proj=down),
                gate=SimpleNamespace(
                    e_score_correction_bias=bias,
                    bias_vl=None,
                    tid2eid=torch.zeros(1) if hash_routed else None,
                ),
            )

        by_logits, by_ids = (
            moe.HopperLowBatchMoE.of(ffn(False)),
            (moe.HopperLowBatchMoE.of(ffn(True))),
        )
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
            want = torch.zeros(m, MOE_HIDDEN, device=DEVICE)
            for k in range(top_k):
                for e in ids[:, k].unique().tolist():
                    rows = (ids[:, k] == e).nonzero().squeeze(1)
                    h = x[rows].float() @ w13[e].t()
                    a = torch.nn.functional.silu(h[:, :MOE_INTER].clamp(max=10.0))
                    a = (a * h[:, MOE_INTER:].clamp(-10.0, 10.0)).bfloat16().float()
                    want[rows] += (a @ w2[e].t()) * weights[rows, k, None]
            stock_error = _rel(stock, want)
            assert stock_error < 6e-2, m

            routed, shared = by_logits(x, router_logits=logits)
            got = _finalize(routed)
            # The draft layers' finalize kernel: the same sum, FMA order.
            assert _rel(finalize(routed), got) < 4e-3, m
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
                stock_shared = apply_humming_linear(
                    down,
                    act(
                        apply_humming_linear(
                            gate_up,
                            x,
                            skip_bias_add=True,
                            layer_config=gate_up_humming.layer_config,
                            compute_config=gate_up_humming.compute_config,
                            locks=gate_up_humming.locks,
                        )
                    ),
                    skip_bias_add=True,
                    layer_config=down_humming.layer_config,
                    compute_config=down_humming.compute_config,
                    locks=down_humming.locks,
                )
                assert _rel(shared, stock_shared) < 1e-2, (
                    m,
                    _rel(shared, stock_shared),
                )
            else:
                assert shared is None


# ------------------------------------------------------------ fused FP8 indexer


def test_fused_indexer_matches_torch_at_long_context():
    """The buffers shared by a ratio-2 and a ratio-1 indexer (rows twice as
    long) serve the ratio-1 op's two requests of 6 verify rows, longer than
    the ratio-2 op's width, the candidate coverage (2048 blocks of 8) and
    three tiles a CTA: the logits rows, written into a poisoned buffer, match
    an FP32 reference position by position (no row writes into its
    neighbour), and the top-512 and candidate blocks match torch.topk and the
    stock candidate select on those logits. A longer step is refused."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper import (
        fused_indexer,
    )

    max_model_len, page, rows_per_req, requests, cand_k = 40000, 64, 6, 2, 2048
    lengths = (36000, 30000)
    rows = rows_per_req * requests
    g = torch.Generator(device=DEVICE).manual_seed(0)
    indexer = fused_indexer.FusedDecodeIndexer.for_ops(
        [
            SimpleNamespace(max_model_len=max_model_len // 2),
            SimpleNamespace(max_model_len=max_model_len),
        ],
        DEVICE,
    )
    assert max(lengths) > max_model_len // 2
    blocks_per_req = -(-max(lengths) // page)
    num_blocks = requests * blocks_per_req + 7
    raw = torch.zeros(num_blocks, page * 132, dtype=torch.uint8, device=DEVICE)
    data = raw[:, : page * 128].view(torch.float8_e4m3fn).view(num_blocks, page, 128)
    scales = raw[:, page * 128 :].view(torch.float32).view(num_blocks, page)
    data.copy_(
        (0.5 * torch.randn(data.shape, generator=g, device=DEVICE)).to(data.dtype)
    )
    scales.copy_(torch.rand(scales.shape, generator=g, device=DEVICE) + 0.5)
    order = torch.randperm(num_blocks, generator=g, device=DEVICE).int()
    table = order[: requests * blocks_per_req].view(requests, blocks_per_req)
    block_table = table.repeat_interleave(rows_per_req, 0).contiguous()
    seq_lens = torch.tensor(
        [n - rows_per_req + 1 + j for n in lengths for j in range(rows_per_req)],
        dtype=torch.int32,
        device=DEVICE,
    )
    q = torch.randn(rows, 32, 128, generator=g, device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, generator=g, device=DEVICE)
    candidates = torch.full((rows, cand_k), -1, dtype=torch.int32, device=DEVICE)
    op = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=raw.view(num_blocks, page, 132)),
        candidate_blocks=candidates,
        candidate_write=True,
        topk_indices_buffer=torch.full(
            (rows, 512), -1, dtype=torch.int32, device=DEVICE
        ),
    )
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            seq_lens=seq_lens,
            block_table=block_table,
            indices=None,
            global_seq_lens=None,
        ),
        num_decode_tokens=rows,
        num_prefills=0,
        max_seq_len=int(seq_lens.max()),
    )
    assert indexer.serves(metadata, rows_per_req)
    assert not indexer.serves(
        SimpleNamespace(**{**vars(metadata), "max_seq_len": indexer.width + 1}),
        rows_per_req,
    )
    indexer.logits.fill_(float("nan"))
    indexer.warm_up(rows_per_req, cand_k)
    indexer(op, q, weights, metadata, rows_per_req)

    pos = torch.arange(indexer.width, device=DEVICE)
    for r in range(rows):
        n = int(seq_lens[r])
        blk = block_table[r, pos[:n] // page].long()
        k = data[blk, pos[:n] % page].float() * scales[blk, pos[:n] % page, None]
        want = (torch.relu(k @ q[r].float().t()) @ weights[r]).float()
        got = indexer.logits[r, :n]
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3 * want.abs().max())
        top = set(op.topk_indices_buffer[r].tolist())
        assert top == set(want.topk(512).indices.tolist()), r
    ref_blocks = torch.full_like(candidates, -1)
    select_candidate_blocks(
        indexer.logits[:rows].clone(), None, seq_lens, cand_k, 8, ref_blocks
    )
    for r in range(rows):
        assert set(candidates[r].tolist()) == set(ref_blocks[r].tolist()), r
