# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 low-batch attention (``specialized_decode/attention``)
against the stock decode path it replaces (the MSA decode indexer with its
top-k, the MSA sparse decode and the MXFP8 o_proj), on random NVFP4 KV and
index caches of one TP rank (16 query heads, one KV head, one index head).

``build_low_batch_attention`` reads a fixed set of the layer's attributes
(its o_proj and the attention constants); a full ``MiniMaxM3SparseAttention``
needs a model config, so the layer here carries those attributes around a
real ``RowParallelLinear`` with MXFP8 weights."""

import math
import os
import tempfile
from contextlib import suppress
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="the low-batch attention kernels run on SM100-family GPUs",
)

HEADS, HEAD_DIM, HIDDEN, PAGE, TOPK, MAX_PAGES = 16, 128, 6144, 128, 16, 8192
Q_SIZE = HEADS * HEAD_DIM
DEVICE = "cuda"


@pytest.fixture(scope="module")
def layer():
    """The layer's attributes around an MXFP8 o_proj with random weights, its
    config and its low-batch attention."""
    from vllm.config import (
        KernelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        build_low_batch_attention,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    quant = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "activation_scheme": "dynamic",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": 16,
            "ignored_layers": [],
        }
    )
    config = VllmConfig(parallel_config=ParallelConfig(), kernel_config=KernelConfig())
    fd, path = tempfile.mkstemp()
    os.close(fd)
    default_dtype = torch.get_default_dtype()
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{path}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            init_workspace_manager(torch.device(DEVICE))
            torch.set_default_dtype(torch.bfloat16)
            o_proj = RowParallelLinear(
                Q_SIZE,
                HIDDEN,
                bias=False,
                quant_config=quant,
                reduce_results=False,
                prefix="layers.0.self_attn.o_proj",
            ).to(DEVICE)
            torch.set_default_dtype(default_dtype)
            generator = torch.Generator(device=DEVICE).manual_seed(0)
            values = torch.randn(
                o_proj.weight.shape, generator=generator, device=DEVICE
            )
            o_proj.weight.data.copy_(
                (values * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
            )
            o_proj.weight_scale.data.copy_(
                torch.randint(114, 118, o_proj.weight_scale.shape, device=DEVICE)
                .to(torch.uint8)
                .view(o_proj.weight_scale.dtype)
            )
            if isinstance(o_proj.quant_method, QuantizeMethodBase):
                o_proj.quant_method.process_weights_after_loading(o_proj)
            attrs = SimpleNamespace(
                o_proj=o_proj,
                num_heads=HEADS,
                num_kv_heads=1,
                num_idx_heads=1,
                head_dim=HEAD_DIM,
                q_size=Q_SIZE,
                kv_cache_dtype="nvfp4",
                impl=SimpleNamespace(scale=HEAD_DIM**-0.5, topk_blocks=TOPK),
                _q_scale_float=0.5,
                _k_scale_float=1.0,
                _v_scale_float=1.0,
                _k_scale=torch.ones((), dtype=torch.float32, device=DEVICE),
                _v_scale=torch.ones((), dtype=torch.float32, device=DEVICE),
            )
            yield attrs, config, build_low_batch_attention(attrs)
    finally:
        torch.set_default_dtype(default_dtype)
        cleanup_dist_env_and_memory()
        with suppress(FileNotFoundError):
            os.remove(path)


def _lengths(context, num_reqs, query_len, generator):
    """Request lengths: ``short`` cycles through one page and the 15- and
    16-history-page edges; ``long`` draws a few thousand pages, with a query
    window straddling two pages and a zero-length padding request last."""
    if context == "short":
        edges = (query_len, 100, 15 * PAGE + 64, 16 * PAGE + 64)
        return [edges[i % 4] for i in range(num_reqs)]
    lengths = (
        torch.randint(1000 * PAGE, 3000 * PAGE, (num_reqs,), generator=generator)
        .add_(1)
        .tolist()
    )
    lengths[0] = lengths[0] // PAGE * PAGE + query_len // 2  # straddles two pages
    if num_reqs > 1:
        lengths[-1] = 0
    return lengths


def _randint(low, high, shape, generator):
    return torch.randint(
        low, high, shape, generator=generator, dtype=torch.uint8, device=DEVICE
    )


def _inputs(context, num_reqs, query_len, seed):
    """Queries, caches, block table and seq_lens of one step. Fifteen pages of
    each request score far above its others (their index keys scaled up), so
    the stock top-k's 16-bit key resolution cannot change the selection;
    every page holds random values of the same scales, so each selected page
    carries about 1/16 of a row's output and a wrong one moves the row by
    about a third."""
    lengths = _lengths(
        context, num_reqs, query_len, torch.Generator().manual_seed(seed)
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = num_reqs * query_len
    pages = [max(math.ceil(n / PAGE), 1) for n in lengths]
    pool = sum(pages) + 4
    perm = torch.randperm(pool, generator=generator, device=DEVICE).to(torch.int32)
    # Unused block table entries point at random pages.
    block_table = torch.randint(
        0, pool, (num_reqs, MAX_PAGES), generator=generator, device=DEVICE
    ).to(torch.int32)
    kv = _randint(0, 256, (pool, 2, PAGE * 72), generator)
    index = _randint(0, 256, (pool, PAGE * 72), generator)
    codes = PAGE * HEAD_DIM // 2
    # E4M3 scales of K, V and the index keys between 0.25 and 0.9.
    kv[:, :, codes:] = _randint(0x28, 0x37, (pool, 2, PAGE * 8), generator)
    index[:, codes:] = _randint(0x28, 0x37, (pool, PAGE * 8), generator)
    cursor = 0
    for b, n in enumerate(lengths):
        phys = perm[cursor : cursor + pages[b]]
        cursor += pages[b]
        block_table[b, : pages[b]] = phys
        history = (max(n, 1) - query_len) // PAGE  # of the request's first row
        if history > TOPK - 1:
            hot = torch.randperm(history, generator=generator, device=DEVICE)
            index[phys[hot[: TOPK - 1]].long(), codes:] = _randint(
                0x58, 0x68, (TOPK - 1, PAGE * 8), generator
            )
    q = torch.randn(rows, Q_SIZE, generator=generator, device=DEVICE) * 2
    iq = torch.randn(rows, HEAD_DIM, generator=generator, device=DEVICE).abs() + 0.1
    return {
        "q": q.to(torch.float8_e4m3fn),
        "iq": iq.to(torch.float8_e4m3fn),
        "kv": kv.view(pool, 2, PAGE, 72),
        "index": index.view(pool, PAGE, 72),
        "block_table": block_table,
        "seq_lens": torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
        "query_len": query_len,
        "live": torch.tensor(lengths).repeat_interleave(query_len) > 0,
    }


def _stock(attrs, inputs):
    """The stock decode path, as the model calls it for a FULL-graph step."""
    from vllm.models.minimax_m3.nvidia import msa_cutlass_sparse_decode as msa
    from vllm.models.minimax_m3.nvidia.indexer_nvfp4 import _msa_indexers

    block_table, seq_lens = inputs["block_table"], inputs["seq_lens"]
    num_reqs, query_len = seq_lens.shape[0], inputs["query_len"]
    rows = num_reqs * query_len
    wrapper_cls = _msa_indexers()[0]
    score = torch.empty(
        wrapper_cls.score_buffer_size(num_reqs, MAX_PAGES, num_heads=1),
        dtype=torch.float32,
        device=DEVICE,
    )
    indexer = wrapper_cls(
        num_heads=1,
        use_cuda_graph=True,
        block_table_buffer=block_table,
        seq_lens_buffer=seq_lens,
        score_buffer=score,
    )
    indexer.plan(block_table, seq_lens, query_len=query_len)
    plan = msa.MSACutlassDecodePlanCache().prepare(
        block_table,
        seq_lens,
        seq_lens.cpu(),
        query_len,
        num_q_heads=HEADS,
        num_kv_heads=1,
        page_size=PAGE,
        topk_blocks=TOPK,
        q8kv4=True,
    )
    topk = torch.empty(
        max(4, (rows + 3) // 4 * 4), 1, TOPK, dtype=torch.int32, device=DEVICE
    )[:rows]
    indexer.run(inputs["iq"].view(-1, 1, HEAD_DIM), inputs["index"], out=topk)
    attn = torch.empty(rows, Q_SIZE, dtype=torch.bfloat16, device=DEVICE)
    msa.msa_cutlass_sparse_decode(
        inputs["q"].view(-1, HEADS, HEAD_DIM),
        inputs["kv"],
        topk,
        attn.view(-1, HEADS, HEAD_DIM),
        plan,
        scale=attrs.impl.scale,
        q_scale_float=attrs._q_scale_float,
        k_scale_float=1.0,
        v_scale_float=1.0,
        k_scale=attrs._k_scale,
        v_scale=attrs._v_scale,
    )
    return attrs.o_proj(attn)[0]


def _check(out, expected, live):
    """Per live row, within the rounding of the two paths: FP8 against F16
    attention probabilities, and the MXFP8 rounding of o_proj's input (about
    4% at most between them). A wrong page moves a row by about a third (see
    ``_inputs``), a wrong mask or scale also far more."""
    for row in torch.nonzero(live).flatten().tolist():
        got, want = out[row].float(), expected[row].float()
        error = float((got - want).norm() / want.norm())
        assert error < 0.06, f"row {row}: relative error {error:.4f}"


def _run(low_batch, inputs):
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        prepare_step,
    )

    step = prepare_step(inputs["block_table"], inputs["seq_lens"], inputs["query_len"])
    return low_batch.run(step, inputs["q"], inputs["iq"], inputs["kv"], inputs["index"])


@pytest.mark.parametrize("context", ["short", "long"])
@pytest.mark.parametrize(
    "num_reqs,query_len", [(1, 1), (4, 1), (32, 1), (1, 8), (4, 8)]
)
def test_low_batch_attention_matches_stock(layer, num_reqs, query_len, context):
    """Eager calls at each request count and query length the decode steps
    take, short contexts (every history page selected) and long ones."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    attrs, config, low_batch = layer
    assert low_batch is not None and low_batch.accepts(num_reqs, query_len)
    inputs = _inputs(context, num_reqs, query_len, seed=num_reqs * 10 + query_len)
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        out = _run(low_batch, inputs)
        expected = _stock(attrs, inputs)
        torch.accelerator.synchronize()
    _check(out, expected, inputs["live"])


@pytest.mark.parametrize("num_reqs,query_len", [(4, 1), (4, 8)])
def test_low_batch_attention_graph_replay(layer, num_reqs, query_len):
    """One CUDA graph captured on one step and replayed on steps with other
    seq_lens, block tables, caches and queries copied into its buffers."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    attrs, config, low_batch = layer
    steps = [
        _inputs(c, num_reqs, query_len, seed)
        for seed, c in enumerate(("long", "short", "long"))
    ]
    pool = max(s["kv"].shape[0] for s in steps)
    static = dict(steps[0])
    static["kv"] = torch.zeros(
        (pool, *steps[0]["kv"].shape[1:]), dtype=torch.uint8, device=DEVICE
    )
    static["index"] = torch.zeros(
        (pool, *steps[0]["index"].shape[1:]), dtype=torch.uint8, device=DEVICE
    )
    static["block_table"] = steps[0]["block_table"].clone()
    static["seq_lens"] = steps[0]["seq_lens"].clone()
    static["q"], static["iq"] = steps[0]["q"].clone(), steps[0]["iq"].clone()

    def load(step):
        for name in ("q", "iq", "block_table", "seq_lens"):
            static[name].copy_(step[name])
        static["kv"][: step["kv"].shape[0]].copy_(step["kv"])
        static["index"][: step["index"].shape[0]].copy_(step["index"])

    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        load(steps[0])
        _run(low_batch, static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _run(low_batch, static)
        for step in steps[1:]:
            load(step)
            graph.replay()
            expected = _stock(attrs, dict(static, live=step["live"]))
            torch.accelerator.synchronize()
            _check(out, expected, step["live"])
