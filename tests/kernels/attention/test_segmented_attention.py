# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Segmented attention accuracy, cache semantics, and startup tuning contracts."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton

pytestmark = pytest.mark.skip_global_cleanup


def _make_paged_attention_case(
    query_lens: list[int],
    context_lens: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    fp8: bool,
    causal: bool = True,
    sliding_window: int = 0,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(419)
    total_queries = sum(query_lens)
    qkv = (
        torch.randn(
            total_queries,
            num_heads + 2 * num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )
    query, key, value = qkv.split((num_heads, num_kv_heads, num_kv_heads), dim=1)
    cache_dtype = current_platform.fp8_dtype() if fp8 else torch.bfloat16
    k_scale = torch.tensor(0.125 if fp8 else 1.0, device=device)
    v_scale = torch.tensor(0.25 if fp8 else 1.0, device=device)
    max_seq_len = max(q + c for q, c in zip(query_lens, context_lens))
    blocks_per_seq = triton.cdiv(max_seq_len, block_size)
    num_blocks = len(query_lens) * blocks_per_seq
    key_cache = (
        torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
        / k_scale
    ).to(cache_dtype)
    value_cache = (
        torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
        / v_scale
    ).to(cache_dtype)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        len(query_lens), blocks_per_seq
    )
    starts = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(
        [q + c for q, c in zip(query_lens, context_lens)],
        device=device,
        dtype=torch.int32,
    )

    first = 0
    for seq, (query_len, context_len) in enumerate(zip(query_lens, context_lens)):
        for query_pos in range(query_len):
            cache_pos = context_len + query_pos
            block = block_table[seq, cache_pos // block_size]
            offset = cache_pos % block_size
            key_cache[block, offset] = (key[first + query_pos].float() / k_scale).to(
                cache_dtype
            )
            value_cache[block, offset] = (
                value[first + query_pos].float() / v_scale
            ).to(cache_dtype)
        first += query_len

    references = []
    first = 0
    for seq, (query_len, context_len) in enumerate(zip(query_lens, context_lens)):
        physical = block_table[seq].long()
        seq_len = context_len + query_len
        full_key = (
            key_cache[physical].flatten(0, 1)[:seq_len].float() * k_scale
        ).repeat_interleave(num_heads // num_kv_heads, dim=1)
        full_value = (
            value_cache[physical].flatten(0, 1)[:seq_len].float() * v_scale
        ).repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", query[first : first + query_len].float(), full_key
        ) / math.sqrt(head_size)
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        query_positions = context_len + torch.arange(query_len, device=device)
        key_positions = torch.arange(seq_len, device=device)
        causal_mask = key_positions[None, :] > query_positions[:, None]
        if causal:
            scores.masked_fill_(causal_mask[None], -float("inf"))
        if sliding_window > 0:
            window_mask = key_positions[None, :] < (
                query_positions[:, None] - sliding_window + 1
            )
            scores.masked_fill_(window_mask[None], -float("inf"))
        if sinks is not None:
            sink_scores = sinks.float()[:, None, None].expand(-1, query_len, 1)
            probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(-1)
            probabilities = probabilities[..., :-1]
        else:
            probabilities = scores.softmax(-1)
        references.append(torch.einsum("hqk,khd->qhd", probabilities, full_value))
        first += query_len

    return {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "starts": starts,
        "seq_lens": seq_lens,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "reference": torch.cat(references),
        "max_seq_len": max_seq_len,
    }


def _run_segmented_case(case, output, max_query_len, *, force_splits=None, **kwargs):
    from vllm.v1.attention.ops.segmented_attention import (
        run_segmented_attention,
        select_segmented_config,
    )

    config = None
    heads, dim = case["query"].shape[1:]
    if force_splits is not None:
        config = dict(
            select_segmented_config(
                case["seq_lens"].numel(),
                max_query_len,
                case["max_seq_len"],
                heads,
                case["key_cache"].shape[2],
                dim,
                case["key_cache"].element_size() == 1,
            ),
            splits=force_splits,
        )
    run_segmented_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        max_query_len,
        case["max_seq_len"],
        case["k_scale"],
        case["v_scale"],
        dim**-0.5,
        config=config,
        **kwargs,
    )


def test_segmented_tuning_candidates_preserve_workspace_bound():
    """D128/D256 candidates retain their existing split scratch bound."""
    from vllm.v1.attention.ops import segmented_attention as segmented
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    for batch, query_len, seq_len, heads, kv_heads, dim, fp8 in (
        (1, 1, 128, 8, 8, 128, True),
        (32, 1, 262144, 16, 1, 128, True),
        (4, 1024, 8192, 16, 4, 256, False),
        (1, 1024, 131072, 6, 1, 256, True),
    ):
        default = segmented.select_segmented_config(
            batch, query_len, seq_len, heads, kv_heads, dim, fp8
        )
        candidates = tuning._candidate_configs(default, batch, query_len, heads, dim)
        assert candidates
        assert (
            tuning._normalized_config(default, batch, query_len, heads, dim)
            in candidates
        )
        assert all(config["splits"] <= default["splits"] for config in candidates)
        assert all(dim % config["bk"] == 0 for config in candidates)


@pytest.mark.parametrize("dim", (128, 256))
def test_segmented_fp8_long_extend_split_workspace_bound(dim):
    from vllm.v1.attention.ops.segmented_attention import (
        MAX_LONG_EXTEND_WORKSPACE_BYTES,
        segmented_query_capacity,
        select_segmented_config,
    )

    for batch, query_len, seq_len, heads in (
        (1, 256, 8192, 6),
        (1, 256, 32768, 6),
        (1, 256, 131072, 6),
        (1, 1024, 32768, 1),
        (1, 1024, 131072, 1),
        (1, 1024, 131072, 6),
        (1, 4096, 131072, 1),
        (4, 1024, 131072, 6),
        (5, 257, 131072, 6),
        (21, 256, 131072, 6),
    ):
        config = select_segmented_config(batch, query_len, seq_len, heads, 1, dim, True)
        if seq_len >= 32768:
            scratch = (
                batch
                * segmented_query_capacity(query_len)
                * heads
                * config["splits"]
                * (dim + 1)
                * 4
            )
            assert scratch <= MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128

    for batch in (1, 4, 16):
        for query_len in (257, 1025, 4095, 8192):
            for ratio in (1, 6, 16):
                for kv_heads in (1, 4):
                    for seq_len in (32768, 131072):
                        heads = ratio * kv_heads
                        config = select_segmented_config(
                            batch, query_len, seq_len, heads, kv_heads, dim, True
                        )
                        if config["splits"] == 1:
                            continue
                        scratch = (
                            batch
                            * segmented_query_capacity(query_len)
                            * heads
                            * config["splits"]
                            * (dim + 1)
                            * 4
                        )
                        assert scratch <= MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128


@torch.inference_mode()
def test_segmented_bf16_d256_long_extend_matches_dense_reference():
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops.segmented_attention import run_segmented_attention

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x segmented prefill")

    case = _make_paged_attention_case(
        [1024],
        [7168],
        num_heads=6,
        num_kv_heads=1,
        head_size=256,
        block_size=1568,
        fp8=False,
    )
    output = torch.empty_like(case["query"])
    run_segmented_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        1024,
        case["max_seq_len"],
        case["k_scale"],
        case["v_scale"],
        256**-0.5,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize(
    "query_len,heads,qk_amplitude,dim",
    [
        (256, 1, 1, 256),
        (256, 6, 1, 256),
        (512, 6, 1, 256),
        (256, 6, 4, 256),
        (256, 6, 16, 256),
        (256, 6, 1, 128),
        (512, 6, 4, 128),
        (256, 6, 16, 128),
        (8192, 1, 1, 128),
        (8192, 1, 1, 256),
    ],
)
@torch.inference_mode()
def test_segmented_fp8_long_extend_matches_dense_reference(
    query_len, heads, qk_amplitude, dim
):
    from vllm.platforms.rocm import on_gfx12x
    from vllm.v1.attention.ops.segmented_attention import run_segmented_attention

    if not current_platform.is_rocm() or not on_gfx12x():
        pytest.skip("FP8 KV requires gfx12")
    case = _make_paged_attention_case(
        [query_len],
        [8192 - query_len],
        num_heads=heads,
        num_kv_heads=1,
        head_size=dim,
        block_size=1568,
        fp8=True,
    )
    if qk_amplitude != 1:
        # Check peaked logits too: one-term FP8 Q quantization passed only the
        # nearly uniform fixture and deviated by 4-70% at these amplitudes.
        case["query"].mul_(qk_amplitude)
        case["k_scale"].mul_(qk_amplitude)
        full_key = (
            case["key_cache"][case["block_table"][0].long()]
            .flatten(0, 1)[:8192]
            .float()
            * case["k_scale"]
        ).repeat_interleave(heads, dim=1)
        full_value = (
            case["value_cache"][case["block_table"][0].long()]
            .flatten(0, 1)[:8192]
            .float()
            * case["v_scale"]
        ).repeat_interleave(heads, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", case["query"].float(), full_key
        ) / math.sqrt(dim)
        future = torch.arange(8192, device=case["query"].device)[None, :] > (
            8192
            - query_len
            + torch.arange(query_len, device=case["query"].device)[:, None]
        )
        scores.masked_fill_(future[None], -float("inf"))
        case["reference"] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_value)
    output = torch.empty_like(case["query"])
    run_segmented_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        query_len,
        case["max_seq_len"],
        case["k_scale"],
        case["v_scale"],
        dim**-0.5,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


def test_segmented_tuning_protects_static_incumbent():
    """Noise-sized gains must not replace the static configuration."""
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    incumbent = {"bm": 16}
    challenger = {"bm": 32}
    samples = {
        tuning._config_key(incumbent): [100.0, 101.0, 99.0, 100.0, 100.0],
        tuning._config_key(challenger): [99.0, 100.0, 98.0, 99.0, 99.0],
    }
    winner, comparisons = tuning._select_tuned_winner(
        incumbent, [incumbent, challenger], samples
    )
    assert winner["config"] == incumbent
    assert comparisons[1]["paired_speedup_vs_default"] < 1.02


def test_segmented_tuning_promotes_verified_challenger():
    """A finalist with a stable material gain should replace the incumbent."""
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    incumbent = {"bm": 16}
    challenger = {"bm": 32}
    samples = {
        tuning._config_key(incumbent): [100.0, 102.0, 98.0, 101.0, 99.0],
        tuning._config_key(challenger): [94.0, 96.0, 92.0, 95.0, 93.0],
    }
    winner, _ = tuning._select_tuned_winner(incumbent, [incumbent, challenger], samples)
    assert winner["config"] == challenger
    assert winner["paired_speedup_vs_default"] > 1.02


def test_segmented_tuning_balances_tp_workloads_without_overlap():
    """Every missing bucket belongs to exactly one reasonably balanced rank."""
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    workloads = list(tuning._workloads(8192, 262144, 32))
    shards = tuning._shard_workloads(
        workloads,
        4,
        8192,
        12,
        2,
        256,
        True,
    )
    flattened = [workload for shard in shards for workload in shard]
    assert sorted(flattened) == sorted(workloads)
    assert len(flattened) == len(set(flattened))
    owners = {
        workload[:2]: rank for rank, shard in enumerate(shards) for workload in shard
    }
    assert all(
        owners[workload[:2]] == rank
        for rank, shard in enumerate(shards)
        for workload in shard
    )

    weights = [
        sum(
            tuning._workload_weight(workload, 8192, 12, 2, 256, True)
            for workload in shard
        )
        for shard in shards
    ]
    largest_group = max(
        sum(
            tuning._workload_weight(workload, 8192, 12, 2, 256, True)
            for workload in workloads
            if workload[:2] == group
        )
        for group in {workload[:2] for workload in workloads}
    )
    assert max(weights) - min(weights) <= largest_group


def test_segmented_tuning_prunes_scheduler_and_kv_limits():
    """Generated buckets must be reachable under scheduler and cache limits."""
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    limits = (8192, 262144, 32)
    raw = set(
        tuning._workloads(
            *limits,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            heads=16,
            kv_heads=1,
            dim=128,
            page=16,
        )
    )
    assert {batch for batch, _, _ in raw} == {1, 2, 4, 8, 16, 32}
    assert {query for _, query, _ in raw} == set(tuning._QUERY_BUCKETS)
    assert (32, 1, 262144) in raw
    assert (1, 8192, 262144) in raw
    assert (1, 8192, 131072) in raw
    assert (1, 8, 131072) in raw
    assert all(
        sum(tuning._query_lengths(batch, query, limits[0])) <= limits[0]
        and batch <= limits[2]
        and query <= min(limits[0], tuning.MAX_QUERY_LEN)
        and query <= seq_len <= limits[1]
        for batch, query, seq_len in raw
    )

    layouts = ((16, 16 * 128 * 2),)
    bounded = set(
        tuning._workloads(
            *limits,
            memory_budget_bytes=2**50,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            heads=16,
            kv_heads=1,
            dim=128,
            page=16,
            cache_layouts=layouts,
            cache_budget_bytes=2**20,
        )
    )
    assert bounded < raw
    assert all(
        batch * sum(((seq_len + block - 1) // block) * size for block, size in layouts)
        <= 2**20
        for batch, _, seq_len in bounded
    )


@pytest.fixture
def segmented_tuner(tmp_path, monkeypatch):
    """Exercise real cache files and lookup while replacing GPU timing only."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops import segmented_attention as segmented
    from vllm.v1.attention.ops import segmented_attention_tuning as tuning

    monkeypatch.setenv("VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", gcnArchName="gfx1201", multi_processor_count=32
        ),
    )
    monkeypatch.setattr(tuning, "_TABLES", {})
    calls = []

    def tune(*args, **kwargs):
        heads, kv_heads, dim = args[3:6]
        batch, query_len, seq_len = args[-1]
        default = segmented.select_segmented_config(
            batch, query_len, seq_len, heads, kv_heads, dim, False
        )
        best = tuning._candidate_configs(default, batch, query_len, heads, dim)[0]
        calls.append(args[-1])
        return {
            "workload": list(args[-1]),
            "query_lengths": tuning._query_lengths(batch, query_len, args[-2]),
            "default": default,
            "best": best,
            "results": [],
        }

    monkeypatch.setattr(tuning, "_tune_workload", tune)
    geometry = (torch.device("cuda:0"), torch.bfloat16, 4, 2, 128, 16, 128**-0.5)
    return SimpleNamespace(
        module=tuning,
        warmup_args=(*geometry, 4, 8, 2),
        lookup_args=(*geometry[:2], torch.bfloat16, *geometry[2:], 1, 1, 8),
        calls=calls,
        cache_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param({}, id="causal"),
        pytest.param({"causal": False}, id="noncausal"),
        pytest.param({"sliding_window": 7}, id="sliding-window"),
        pytest.param({"sliding_window": 7, "causal": False}, id="draft-window"),
        pytest.param({"has_sinks": True}, id="sinks"),
    ],
)
def test_segmented_tuning_persists_without_retuning(segmented_tuner, monkeypatch, mode):
    """A fresh in-memory table reloads the same winner without timing or writing."""
    state = segmented_tuner
    tuning = state.module
    tuning.warmup_segmented_attention(
        *state.warmup_args, memory_budget_bytes=2**50, **mode
    )
    assert state.calls
    expected = tuning.get_segmented_config(*state.lookup_args, **mode)
    assert expected is not None
    (cache,) = state.cache_dir.rglob("*.json")
    saved = cache.read_bytes(), cache.stat().st_mtime_ns
    tuning._TABLES.clear()

    def forbidden(*args, **kwargs):
        pytest.fail("Persistent segmented startup cache attempted to retune")

    monkeypatch.setattr(tuning, "_tune_workload", forbidden)
    tuning.warmup_segmented_attention(
        *state.warmup_args, memory_budget_bytes=2**50, **mode
    )
    assert tuning.get_segmented_config(*state.lookup_args, **mode) == expected
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == saved


@pytest.mark.parametrize(
    "max_tokens,mode",
    [
        pytest.param(3, {}, id="scheduler-token-budget"),
        pytest.param(4, {"causal": False}, id="noncausal"),
        pytest.param(4, {"sliding_window": 7}, id="sliding-window"),
        pytest.param(4, {"sliding_window": 7, "causal": False}, id="draft-window"),
        pytest.param(4, {"has_sinks": True}, id="sinks"),
    ],
)
def test_segmented_tuning_does_not_reuse_incompatible_winners(
    segmented_tuner, max_tokens, mode
):
    """Changed attention semantics or query mix require their own measurements."""
    state = segmented_tuner
    tuning = state.module
    tuning.warmup_segmented_attention(*state.warmup_args, memory_budget_bytes=2**50)
    previous_calls = len(state.calls)
    if mode:
        assert tuning.get_segmented_config(*state.lookup_args, **mode) is None
    args = (*state.warmup_args[:7], max_tokens, *state.warmup_args[8:])
    tuning.warmup_segmented_attention(*args, memory_budget_bytes=2**50, **mode)
    assert len(state.calls) > previous_calls
    assert len(list(state.cache_dir.rglob("*.json"))) == 2
    assert tuning.get_segmented_config(*state.lookup_args, **mode) is not None


@pytest.mark.parametrize(
    "dim,page,hq,hk,dtype,kv_dtype,byte_cache,force_splits",
    [
        (128, 16, 4, 4, torch.bfloat16, torch.bfloat16, False, None),
        (128, 32, 8, 2, torch.float16, torch.float16, False, None),
        (256, 784, 12, 2, torch.bfloat16, torch.bfloat16, False, None),
        (256, 1568, 12, 2, torch.bfloat16, torch.float8_e4m3fn, False, None),
        (128, 32, 16, 1, torch.bfloat16, torch.float8_e4m3fn, True, None),
        (256, 32, 4, 2, torch.float16, torch.float8_e4m3fn, False, None),
        (128, 32, 8, 2, torch.bfloat16, torch.bfloat16, False, 32),
    ],
)
@torch.inference_mode()
def test_segmented_attention_ragged_unified_graph_replay(
    monkeypatch, dim, page, hq, hk, dtype, kv_dtype, byte_cache, force_splits
):
    """Unified-cache segmented prefill follows metadata on graph replay."""
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

    if not current_platform.is_rocm() or not (
        on_gfx12x() if kv_dtype.itemsize == 1 else on_gfx1x()
    ):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    if force_splits is not None:
        original = dispatcher.select_segmented_config
        monkeypatch.setattr(
            dispatcher,
            "select_segmented_config",
            lambda *args: dict(original(*args), splits=force_splits),
        )
    query_lengths = [0, 1, 2, 7, 33, 129]
    contexts = [0, 67, 0, page - 1, page + 1, 33]
    device = torch.device("cuda:0")
    total = sum(query_lengths)
    q = torch.randn(total, hq + 1, dim, device=device, dtype=dtype)[:, :hq]
    k = torch.randn(total, hk + 1, dim, device=device, dtype=dtype)[:, :hk]
    v = torch.randn_like(k)
    blocks_per_seq = triton.cdiv(max(map(sum, zip(query_lengths, contexts))), page)
    num_blocks = len(contexts) * blocks_per_seq
    page_elements = page * hk * dim
    backing = torch.empty(num_blocks * 2 * page_elements, device=device, dtype=kv_dtype)
    cache_shape = (num_blocks, page, hk, dim)
    strides = (2 * page_elements, hk * dim, dim, 1)
    kc = torch.as_strided(backing, cache_shape, strides)
    vc = torch.as_strided(backing, cache_shape, strides, page_elements)
    ks = torch.tensor(0.13 if kv_dtype.itemsize == 1 else 1.0, device=device)
    vs = torch.tensor(0.27 if kv_dtype.itemsize == 1 else 1.0, device=device)
    kc.copy_((torch.randn(cache_shape, device=device).float() * 0.25 / ks).to(kv_dtype))
    vc.copy_((torch.randn(cache_shape, device=device).float() * 0.25 / vs).to(kv_dtype))
    table = torch.randperm(num_blocks, device=device, dtype=torch.int32).view(
        len(contexts), blocks_per_seq
    )
    host_table = table.cpu().tolist()
    starts = torch.empty(len(contexts) + 1, device=device, dtype=torch.int32)
    lengths = torch.empty(len(contexts), device=device, dtype=torch.int32)
    output = torch.full((total, hq + 1, dim), 17.0, device=device, dtype=dtype)[:, :hq]

    def metadata(qlens):
        starts.copy_(
            torch.tensor(
                [0, *torch.tensor(qlens).cumsum(0).tolist()],
                device=device,
                dtype=torch.int32,
            )
        )
        lengths.copy_(
            torch.tensor(
                [c + n for c, n in zip(contexts, qlens)],
                device=device,
                dtype=torch.int32,
            )
        )
        first = 0
        for seq, (nq, context) in enumerate(zip(qlens, contexts)):
            for local in range(nq):
                position = context + local
                physical = host_table[seq][position // page]
                offset = position % page
                kc[physical, offset].copy_((k[first + local].float() / ks).to(kv_dtype))
                vc[physical, offset].copy_((v[first + local].float() / vs).to(kv_dtype))
            first += nq

    def run():
        dispatcher.segmented_attention(
            query=q,
            key=k,
            value=v,
            output=output,
            kv_cache_dtype="fp8" if kv_dtype.itemsize == 1 else "auto",
            key_cache=kc.view(torch.uint8) if byte_cache else kc,
            value_cache=vc.view(torch.uint8) if byte_cache else vc,
            block_table=table,
            query_start_loc=starts,
            seq_lens=lengths,
            max_seq_len=max(contexts) + 129,
            max_query_len=129,
            k_scale=ks,
            v_scale=vs,
            sm_scale=dim**-0.5,
        )

    def check(qlens):
        first = 0
        for seq, (nq, context) in enumerate(zip(qlens, contexts)):
            if nq == 0:
                first += nq
                continue
            physical = table[seq].long()
            full_k = (
                kc[physical].reshape(-1, hk, dim)[: context + nq].float() * ks
            ).repeat_interleave(hq // hk, 1)
            full_v = (
                vc[physical].reshape(-1, hk, dim)[: context + nq].float() * vs
            ).repeat_interleave(hq // hk, 1)
            logits = torch.einsum(
                "qhd,khd->hqk", q[first : first + nq].float(), full_k
            ) / math.sqrt(dim)
            mask = (
                torch.arange(context + nq, device=device)[None, :]
                > context + torch.arange(nq, device=device)[:, None]
            )
            logits.masked_fill_(mask[None], -float("inf"))
            ref = torch.einsum("hqk,khd->qhd", logits.softmax(-1), full_v)
            actual = output[first : first + nq].float()
            assert torch.isfinite(actual).all()
            relative = (actual - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
            assert relative.max().item() < 0.01
            first += nq

    metadata(query_lengths)
    run()
    torch.accelerator.synchronize()
    check(query_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replay_lengths = [1, 0, 7, 2, 34, 128]
    q.mul_(0.75)
    k.add_(0.25)
    v.add_(0.5)
    metadata(replay_lengths)
    output.fill_(17.0)
    graph.replay()
    torch.accelerator.synchronize()
    check(replay_lengths)


@pytest.mark.parametrize(
    "fp8,force_splits",
    [(False, None), (True, None), (False, 4), (True, 4)],
)
@torch.inference_mode()
def test_segmented_ragged_prefill_matches_dense_reference(fp8, force_splits):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    case = _make_paged_attention_case(
        [1, 2, 7, 129],
        [31, 64, 97, 128],
        num_heads=12,
        num_kv_heads=2,
        head_size=256,
        block_size=32,
        fp8=fp8,
    )
    output = torch.empty_like(case["query"])
    _run_segmented_case(case, output, max_query_len=129, force_splits=force_splits)

    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_rocm_segmented_attn_layout_cache_update(fp8):
    from types import SimpleNamespace

    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_segmented_attn import (
        RocmSegmentedAttentionImpl,
    )

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    kv_cache_dtype = "fp8" if fp8 else "auto"
    impl = RocmSegmentedAttentionImpl(
        12,
        256,
        256**-0.5,
        2,
        None,
        None,
        kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )
    cache = torch.zeros(3, 2, 32, 512, device="cuda", dtype=cache_dtype)
    key = torch.randn(5, 2, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 7, 32, 65, 95], device="cuda", dtype=torch.int64)
    k_scale = torch.tensor(0.125 if fp8 else 1.0, device="cuda")
    v_scale = torch.tensor(0.25 if fp8 else 1.0, device="cuda")
    layer = SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)

    impl.do_kv_cache_update(layer, key, value, cache, slots)
    key_cache, value_cache = impl._split_kv_cache(cache)
    if fp8:
        key_cache = key_cache.view(impl.fp8_dtype)
        value_cache = value_cache.view(impl.fp8_dtype)
    cached_key = torch.stack(
        [key_cache[int(slot) // 32, int(slot) % 32] for slot in slots]
    )
    cached_value = torch.stack(
        [value_cache[int(slot) // 32, int(slot) % 32] for slot in slots]
    )
    expected_key = (key.float() / k_scale).to(key_cache.dtype)
    expected_value = (value.float() / v_scale).to(value_cache.dtype)
    torch.testing.assert_close(cached_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(cached_value, expected_value, rtol=0, atol=0)


@torch.inference_mode()
def test_segmented_mixed_causality_matches_dense_reference():
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x unified attention per-request causal fallback")

    query_lens = [2, 3]
    context_lens = [5, 6]
    args = dict(num_heads=8, num_kv_heads=2, head_size=128, block_size=32)
    case = _make_paged_attention_case(
        query_lens, context_lens, fp8=False, causal=True, **args
    )
    noncausal_case = _make_paged_attention_case(
        query_lens, context_lens, fp8=False, causal=False, **args
    )
    reference = torch.cat(
        (
            case["reference"][: query_lens[0]],
            noncausal_case["reference"][query_lens[0] :],
        )
    )
    causal = torch.tensor([True, False], device=case["query"].device)
    output = torch.empty_like(case["query"])
    dispatcher.segmented_attention(
        query=case["query"],
        key=case["key"],
        value=case["value"],
        output=output,
        kv_cache_dtype="auto",
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        block_table=case["block_table"],
        query_start_loc=case["starts"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        max_query_len=max(query_lens),
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        sm_scale=128**-0.5,
        causal=causal,
    )

    relative = (output.float() - reference).norm(dim=-1) / reference.norm(
        dim=-1
    ).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window_size", [1, 2048])
@torch.inference_mode()
def test_segmented_sliding_window_matches_dense_reference(fp8, causal, window_size):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    case = _make_paged_attention_case(
        [8],
        [4096],
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=32,
        fp8=fp8,
        causal=causal,
        sliding_window=window_size,
    )
    output = torch.empty_like(case["query"])
    _run_segmented_case(
        case, output, max_query_len=8, sliding_window=window_size - 1, causal=causal
    )

    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("window_size", [0, 128])
@pytest.mark.parametrize("head_size", [64, 128])
@torch.inference_mode()
def test_segmented_sinks_match_dense_reference(fp8, window_size, head_size):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    sinks = torch.linspace(
        -1.0,
        8.0,
        16,
        device="cuda:0",
        dtype=torch.bfloat16 if head_size == 64 else torch.float32,
    )
    case = _make_paged_attention_case(
        [1, 7],
        [127, 130],
        num_heads=16,
        num_kv_heads=2,
        head_size=head_size,
        block_size=32,
        fp8=fp8,
        sliding_window=window_size,
        sinks=sinks,
    )
    output = torch.empty_like(case["query"])
    _run_segmented_case(
        case,
        output,
        max_query_len=7,
        sliding_window=window_size - 1 if window_size else -1,
        sinks=sinks,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("window_size", [0, 128])
@pytest.mark.parametrize("query_len", [1, 8])
@torch.inference_mode()
def test_segmented_sink_split_reduce_matches_reference(fp8, window_size, query_len):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops.segmented_attention import (
        run_segmented_attention,
        select_segmented_config,
    )

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    sinks = torch.linspace(-1.0, 8.0, 16, device="cuda:0")
    case = _make_paged_attention_case(
        [query_len],
        [8192 - query_len],
        num_heads=16,
        num_kv_heads=2,
        head_size=64,
        block_size=64,
        fp8=fp8,
        sliding_window=window_size,
        sinks=sinks,
    )
    attention_span = min(8192, window_size + query_len) if window_size else 8192
    config = dict(
        select_segmented_config(1, query_len, attention_span, 16, 2, 64, fp8),
        splits=4,
    )
    output = torch.empty_like(case["query"])
    run_segmented_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        query_len,
        8192,
        case["k_scale"],
        case["v_scale"],
        64**-0.5,
        sliding_window=window_size - 1 if window_size else -1,
        sinks=sinks,
        config=config,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("query_len,context_len", [(1, 8191), (8, 32760), (1, 131071)])
@torch.inference_mode()
def test_segmented_long_context_fp8_sinks_match_dense_reference(query_len, context_len):
    from vllm.platforms.rocm import on_gfx12x

    if not current_platform.is_rocm() or not on_gfx12x():
        pytest.skip("FP8 KV requires gfx12")
    sinks = torch.linspace(-1.0, 1.0, 16, device="cuda:0", dtype=torch.bfloat16)
    case = _make_paged_attention_case(
        [query_len],
        [context_len],
        num_heads=16,
        num_kv_heads=2,
        head_size=64,
        block_size=64,
        fp8=True,
        sinks=sinks,
    )
    output = torch.empty_like(case["query"])
    _run_segmented_case(case, output, max_query_len=query_len, sinks=sinks)
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize(
    "feature",
    [
        "softcap",
        "sliding_softcap",
        "output_scale",
    ],
)
@torch.inference_mode()
def test_segmented_softcap_and_output_scaling_match_dense_reference(feature):
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x unified attention feature fallback")

    num_heads = 8
    configured_sliding_window = 16 if feature.startswith("sliding") else 0
    sliding_window = configured_sliding_window - 1 if configured_sliding_window else -1
    softcap = 5.0 if "softcap" in feature else 0.0
    case = _make_paged_attention_case(
        [7, 2],
        [40, 35],
        num_heads=num_heads,
        num_kv_heads=2,
        head_size=128,
        block_size=32,
        fp8=False,
        sliding_window=configured_sliding_window,
        softcap=softcap,
        sinks=None,
    )
    output_scale = None
    if feature == "output_scale":
        output_scale = torch.tensor(0.5, device="cuda:0")
        output = torch.empty_like(case["query"], dtype=current_platform.fp8_dtype())
    else:
        output = torch.empty_like(case["query"])

    dispatcher.segmented_attention(
        query=case["query"],
        key=case["key"],
        value=case["value"],
        output=output,
        kv_cache_dtype="auto",
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        block_table=case["block_table"],
        query_start_loc=case["starts"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        max_query_len=7,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        sm_scale=128**-0.5,
        sliding_window=sliding_window,
        softcap=softcap,
        output_scale=output_scale,
        sinks=None,
    )

    actual = output.float()
    if output_scale is not None:
        actual *= output_scale
        torch.testing.assert_close(actual, case["reference"], atol=0.2, rtol=0.2)
    else:
        relative = (actual - case["reference"]).norm(dim=-1) / case["reference"].norm(
            dim=-1
        ).clamp_min(1e-6)
        assert torch.isfinite(output).all() and relative.max().item() < 0.01


@torch.inference_mode()
def test_segmented_attention_cache_offsets_cross_int32_boundary():
    """A small logical prefix can live beyond 2**31 elements in a strided cache."""
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops.segmented_attention import run_segmented_attention

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x segmented prefill")
    torch.accelerator.empty_cache()
    free, _ = torch.accelerator.get_memory_info()
    if free < 10 * 2**30:
        pytest.skip("The address-boundary fixture needs 10 GiB free VRAM")
    device = torch.device("cuda:0")
    stride = 2**30 + 4096
    kc = torch.empty_strided(
        (3, 32, 1, 128),
        (stride, 128, 128, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_strided(
        (3, 32, 1, 128),
        (stride, 128, 128, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_k = torch.randn(2, 32, 1, 128, device=device, dtype=torch.bfloat16)
    dense_v = torch.randn_like(dense_k)
    for logical, physical in enumerate((2, 1)):
        kc[physical].copy_(dense_k[logical])
        vc[physical].copy_(dense_v[logical])
    q = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    table = torch.tensor([[2, 1]], device=device, dtype=torch.int32)
    starts = torch.tensor([0, 2], device=device, dtype=torch.int32)
    lengths = torch.tensor([35], device=device, dtype=torch.int32)
    one = torch.ones((), device=device)
    run_segmented_attention(
        q, out, kc, vc, table, starts, lengths, 2, 35, one, one, 128**-0.5
    )
    full_k = dense_k.flatten(0, 1)[:35].float().repeat_interleave(4, 1)
    full_v = dense_v.flatten(0, 1)[:35].float().repeat_interleave(4, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), full_k) / math.sqrt(128)
    scores[:, 0, -1] = -float("inf")
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_v)
    error = ((out.float() - expected).norm(dim=-1) / expected.norm(dim=-1)).max()
    assert torch.isfinite(out).all() and error < 0.01
