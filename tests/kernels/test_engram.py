# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.common import engram as engram_ops
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
)
from vllm.platforms import current_platform


def _reference_engram_post_wkv(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    eps,
    clamp_value,
    kv_start,
):
    num_tokens, hc_mult, dim = hidden_states.shape
    source = torch.arange(num_tokens, device=kv.device) + kv_start
    valid = source < kv.shape[0]
    local_kv = kv.new_zeros(num_tokens, kv.shape[1])
    local_kv[valid] = kv[source[valid]]

    key = local_kv[:, : hc_mult * dim].float().view(num_tokens, hc_mult, dim)
    value = local_kv[:, hc_mult * dim :].float()
    h = hidden_states.float()
    weight = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        local_mask = torch.zeros(num_tokens, dtype=torch.bool, device=kv.device)
        local_mask[valid] = token_mask[source[valid]]
        gate = gate.masked_fill(~local_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(hidden_states.dtype)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "num_tokens,num_kv_tokens,hc_mult,dim,tp_size,tp_rank,use_mask",
    [
        (1, 1, 4, 5120, 1, 0, False),
        (3, 5, 4, 96, 2, 0, True),
        (3, 5, 2, 257, 2, 1, False),
        (3, 5, 4, 96, 2, 1, True),
        (1, 1, 4, 96, 4, 3, True),
        (0, 0, 4, 96, 4, 0, False),
    ],
)
def test_fused_engram_post_wkv_matches_reference(
    num_tokens,
    num_kv_tokens,
    hc_mult,
    dim,
    tp_size,
    tp_rank,
    use_mask,
    monkeypatch,
):
    """Preserve gate math, WKV layout, masking, and SP padding semantics."""
    torch.manual_seed(0)
    device = "cuda"
    hidden_states = torch.randn(
        num_tokens, hc_mult, dim, dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        num_kv_tokens,
        (hc_mult + 1) * dim,
        dtype=torch.bfloat16,
        device=device,
    )
    q_weight = torch.randn(hc_mult, dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(hc_mult, dim, dtype=torch.bfloat16, device=device)
    token_mask = None
    if use_mask:
        token_mask = torch.arange(num_kv_tokens, device=device) % 2 == 0
    eps = 1e-20
    clamp_value = 1e-6
    use_sequence_parallel = tp_size > 1
    shard_size = (num_kv_tokens + tp_size - 1) // tp_size
    kv_start = tp_rank * shard_size if use_sequence_parallel else 0

    expected = _reference_engram_post_wkv(
        hidden_states,
        kv,
        q_weight,
        k_weight,
        token_mask,
        eps,
        clamp_value,
        kv_start,
    )
    monkeypatch.setattr(
        engram_ops, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_rank", lambda: tp_rank)

    module = Engram.__new__(Engram)
    torch.nn.Module.__init__(module)
    module.dim = dim
    module.hc_mult = hc_mult
    module.eps = eps
    module.clamp_value = clamp_value
    module.use_sequence_parallel = use_sequence_parallel
    module.embed_tokens = torch.nn.Identity()
    # `forward` reads rows staged by `prepare_embeddings`, so inject kv there.
    module.embed_tokens.tp_size = 1
    module.staged_rows = kv.unsqueeze(1)
    if use_sequence_parallel:
        padded = torch.nn.functional.pad(kv, (0, 0, 0, (-num_kv_tokens) % tp_size))
        module.staged_rows = padded[
            tp_rank * shard_size : (tp_rank + 1) * shard_size
        ].unsqueeze(1)
    module.wkv = torch.nn.Identity()

    def check_local_projection(_, args):
        assert args[0].shape[0] == num_tokens

    module.wkv.register_forward_pre_hook(check_local_projection)
    module.q_weight = torch.nn.Parameter(q_weight, requires_grad=False)
    module.k_weight = torch.nn.Parameter(k_weight, requires_grad=False)

    actual = module(hidden_states, kv.unsqueeze(1), token_mask)

    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def _hash_state(use_slot_cache: bool) -> NgramHashState:
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.block_size = 64
    state.use_slot_cache = use_slot_cache
    state._cache = None
    state._kv_cache_ref = None
    state.swa_cache_module = torch.nn.Module()
    state.swa_cache_module.kv_cache = torch.empty(0)
    return state


@pytest.mark.parametrize("num_blocks", [4, 100])
def test_engram_cache_follows_kv_cache_binding(num_blocks):
    """Discard profiling history when the real KV cache replaces its storage."""
    state = _hash_state(use_slot_cache=True)
    assert not state.ensure_cache()

    state.swa_cache_module.kv_cache = torch.empty(4, 64, 1)
    assert state.ensure_cache()
    profiling_cache = state._cache
    profiling_cache.fill_(7)
    assert state.ensure_cache()
    assert state._cache is profiling_cache
    assert torch.all(state._cache == 7)

    state.swa_cache_module.kv_cache = torch.empty(num_blocks, 64, 1)
    assert state.ensure_cache()
    assert state._cache is not profiling_cache
    assert state._cache.shape == (num_blocks * 64,)
    assert torch.count_nonzero(state._cache) == 0

    state.swa_cache_module.kv_cache = torch.empty(0)
    assert not state.ensure_cache()
    assert state._cache is None


def test_engram_without_slot_cache_only_gates_on_kv_binding():
    """The V2 runner supplies every lookback, so no slot cache is allocated."""
    state = _hash_state(use_slot_cache=False)
    assert not state.ensure_cache()
    state.swa_cache_module.kv_cache = torch.empty(4, 64, 1)
    assert state.ensure_cache()
    assert state._cache is None


def _reference_hashes(
    ids,
    positions,
    slots,
    starts,
    tables,
    dead,
    cache,
    token_map,
    multipliers,
    primes,
    offsets,
):
    """Scalar n-gram oracle with persistent, physically addressed token history."""
    for token, slot, is_dead in zip(ids, slots, dead):
        if slot >= 0:
            cache[slot] = -1 if is_dead else token_map[token]
    result = []
    for i, position in enumerate(positions):
        req = min(bisect.bisect_right(starts[1:], i), len(tables) - 1)
        history, blocked = [], False
        for shift in range(4):
            lookback = position - shift
            p = max(0, min(lookback, len(tables[req]) * 16 - 1))
            source = cache[tables[req][p // 16] * 16 + p % 16]
            blocked |= lookback < 0 or source == -1
            history.append(0 if blocked else source)
        layers = []
        for layer, row in enumerate(multipliers):
            rolling, hashes = history[0] * row[0], []
            for j, head_primes in enumerate(primes[layer], start=1):
                rolling ^= history[j] * row[j]
                for head, prime in enumerate(head_primes):
                    col = (j - 1) * len(head_primes) + head
                    hashes.append(rolling % prime + offsets[layer][col])
            layers.append(hashes)
        result.append(layers)
    return torch.tensor(result, dtype=torch.int32)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("query_len", [1, 17, 257])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided_slots", [False, True])
@pytest.mark.parametrize("large_hashes", [False, True])
def test_engram_hash_cache_replay(query_len, capture, strided_slots, large_hashes):
    """Preserve history, dead tokens and padding as inputs change across replay."""
    num_tokens = 2 * query_len + 3
    blocks = (3 * query_len + 15) // 16
    tables = [list(range(blocks, 2 * blocks)), list(range(blocks))]
    starts = [0, query_len, 2 * query_len]

    def tensor(data, dtype=torch.int64):
        return torch.tensor(data, dtype=dtype, device="cuda")

    input_ids = tensor([0] * num_tokens)
    positions = torch.zeros_like(input_ids)
    slots = tensor([-1] * (num_tokens * (2 if strided_slots else 1)))
    if strided_slots:
        slots = slots[::2]
    dead_mask = tensor([False] * num_tokens, torch.bool)
    cache = tensor([-1] * (2 * blocks * 16), torch.int32)
    token_map = [i * 3091 if large_hashes else i % 7 for i in range(32)]
    multipliers = (
        engram_ops.compute_hash_multipliers((1, 14), 4, 99092).tolist()
        if large_hashes
        else [[3, 5, 7, 9], [11, 13, 15, 17]]
    )
    primes = (
        [
            [
                [16000057 + 2 * (layer * 24 + j * 8 + h) for h in range(8)]
                for j in range(3)
            ]
            for layer in range(2)
        ]
        if large_hashes
        else [[[19], [23], [29]]] * 2
    )
    offsets = []
    for layer in primes:
        row, offset = [], 0
        for heads in layer:
            for prime in heads:
                row.append(offset)
                offset += prime
        offsets.append(row)
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map = tensor(token_map, torch.int32)
    state.multipliers = tensor(multipliers)
    state.primes = tensor(primes)
    state.offsets = tensor(offsets)
    state._cache = cache
    state.use_slot_cache = True
    state.pad_id = 0
    state.block_size = 16
    lookback = tensor([[-1] * 3] * 2, torch.int32)
    args = (
        input_ids,
        positions,
        tensor(starts, torch.int32),
        dead_mask,
        lookback,
        torch.zeros_like(lookback, dtype=torch.bool),
        slots,
        tensor(tables, torch.int32),
    )
    op = state
    graph = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = op(*args)
        torch.cuda.current_stream().wait_stream(stream)
    cache.fill_(-1)
    expected_cache = [-1] * cache.numel()

    for step in range(3):
        ids = [(i + 3 * step) % 32 for i in range(num_tokens)]
        pos = list(range(step * query_len, (step + 1) * query_len)) * 2
        physical_slots = [
            tables[i // query_len][p // 16] * 16 + p % 16 for i, p in enumerate(pos)
        ]
        pos += [0] * 3
        physical_slots += [-1] * 3
        # A fully padded replay must leave every cache slot untouched.
        if step == 2:
            physical_slots = [-1] * num_tokens
        dead = [(i + step) % 5 == 0 for i in range(num_tokens)]
        input_ids.copy_(tensor(ids))
        positions.copy_(tensor(pos))
        slots.copy_(tensor(physical_slots))
        dead_mask.copy_(tensor(dead, torch.bool))
        if graph is None:
            output = op(*args)
        else:
            graph.replay()
        expected = _reference_hashes(
            ids,
            pos,
            physical_slots,
            starts,
            tables,
            dead,
            expected_cache,
            token_map,
            multipliers,
            primes,
            offsets,
        )
        if step != 2:
            real = 2 * query_len
            torch.testing.assert_close(
                output[:real].cpu(), expected[:real], rtol=0, atol=0
            )
        assert cache.cpu().tolist() == expected_cache

    empty_output = state(
        input_ids[:0], positions[:0], args[2], dead_mask[:0], *args[4:]
    )
    assert empty_output.shape == (0, 2, len(offsets[0]))
    assert cache.cpu().tolist() == expected_cache


def _hash_ids(requests, tables_meta, block_table, cache=None, capture=False):
    """Run the op on a batch of (token_ids, start, window) requests; `window`
    is the runner's [depth] lookback for that request (None = all unknown).
    Token IDs 14 and 22 stand in for image tokens that break n-grams.
    Returns one [len(token_ids), ...] hash tensor per request."""
    token_map, multipliers, primes, offsets, block_size = tables_meta
    depth = multipliers.shape[1] - 1

    def tensor(data, dtype=torch.int64):
        return torch.tensor(data, dtype=dtype, device="cuda")

    input_ids, positions, slots, windows, starts = [], [], [], [], [0]
    for req, (token_ids, start, window) in enumerate(requests):
        pos = list(range(start, start + len(token_ids)))
        input_ids += token_ids
        positions += pos
        slots += [
            int(block_table[req, p // block_size]) * block_size + p % block_size
            for p in pos
        ]
        windows.append(window or [-1] * depth)
        starts.append(starts[-1] + len(token_ids))
    window = tensor(windows, torch.int32)
    state = _hash_state(use_slot_cache=cache is not None)
    state.token_map = token_map
    state.multipliers = multipliers
    state.primes = primes
    state.offsets = offsets
    state.pad_id = 0
    state.block_size = block_size
    state._cache = cache
    input_tensor = tensor(input_ids, torch.int32)
    args = (
        input_tensor,
        tensor(positions),
        tensor(starts, torch.int32),
        (input_tensor == 14) | (input_tensor == 22),
        window,
        (window == 14) | (window == 22),
        tensor(slots) if cache is not None else None,
        block_table if cache is not None else None,
    )
    out = state(*args)
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = state(*args)
        graph.replay()
    return [out[a:b] for a, b in zip(starts, starts[1:])]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("runner", ["v2_window_only", "v1_prompt_window_plus_cache"])
@pytest.mark.parametrize("capture", [False, True])
def test_lookback_window_reproduces_single_instance(runner, capture):
    """Decoding on a fresh instance whose prompt KV came from elsewhere (P/D,
    offload) must hash like the instance that processed the whole sequence,
    with two requests batched and chunks of 1-2 tokens so in-batch, window
    and (V1) cache lookbacks mix within one step.

    V2 supplies every lookback from its device token history and needs no
    slot cache. V1 supplies prompt positions only and reads generated
    positions from the slot cache it fills itself. Without any window the
    first decode token hashes stale slots (negative control)."""
    block_size = 4
    block_table = torch.tensor(
        [[3, 1, 5, 0], [2, 4, 6, 7]], dtype=torch.int32, device="cuda"
    )
    tables_meta = (
        torch.arange(50, dtype=torch.int32, device="cuda"),
        torch.tensor([[3, 5, 7, 9]], device="cuda"),
        torch.tensor([[[97, 89], [83, 79], [73, 71]]], device="cuda"),
        torch.tensor([[0, 97, 186, 269, 348, 421]], device="cuda"),
        block_size,
    )
    depth = 3
    prompts = [[11, 12, 13, 14, 15, 16], [31, 32, 33, 34, 35]]
    decodes = [[21, 22, 23, 24], [41, 42, 43, 44]]
    chunk_sizes = [1, 2, 1]
    v2 = runner.startswith("v2")

    def fresh_cache():
        return torch.zeros(8 * block_size, dtype=torch.int32, device="cuda")

    reference = _hash_ids(
        [(p + d, 0, None) for p, d in zip(prompts, decodes)],
        tables_meta,
        block_table,
        cache=fresh_cache(),
    )

    histories = [list(p) for p in prompts]
    cache = None if v2 else fresh_cache()
    consumed = 0
    for chunk in chunk_sizes:
        batch = []
        for history, prompt, decode in zip(histories, prompts, decodes):
            start = len(history)
            tokens = decode[consumed : consumed + chunk]
            window = [
                history[start - 1 - j] if v2 or start - 1 - j < len(prompt) else -1
                for j in range(depth)
            ]
            batch.append((tokens, start, window))
        got = _hash_ids(batch, tables_meta, block_table, cache=cache, capture=capture)
        for req, (tokens, start, _) in enumerate(batch):
            assert torch.equal(got[req], reference[req][start : start + chunk]), (
                f"{runner} request {req} chunk at {start}"
            )
            histories[req] += tokens
        consumed += chunk

    unseeded = _hash_ids(
        [(decodes[0][:1], len(prompts[0]), None)],
        tables_meta,
        block_table,
        cache=fresh_cache(),
    )
    assert not torch.equal(unseeded[0][0], reference[0][len(prompts[0])])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_v2_model_state_gathers_lookback_window():
    """The window is gathered on device from the runner's token history."""
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    depth, max_num_reqs, max_model_len = 3, 4, 16
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.rope_state = None
    state.supports_mm_inputs = False
    state.prompt_embeds_state = None
    state.lookback_token_ids = torch.full(
        (max_num_reqs, depth), -1, dtype=torch.int32, device="cuda"
    )

    all_token_ids = torch.zeros(max_num_reqs, max_model_len, dtype=torch.int32)
    all_token_ids[2, :8] = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22])
    all_token_ids[0, :5] = torch.arange(1, 6)
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([0, 9, 8, 6], dtype=torch.int32, device="cuda")
        ),
        all_token_ids=SimpleNamespace(gpu=all_token_ids.cuda()),
    )
    # Batch rows -> request state rows: a request two decode steps in, a fresh
    # request, and one at the end of a 5-token prompt.
    input_batch = SimpleNamespace(
        idx_mapping=torch.tensor([2, 0, 3], dtype=torch.int64, device="cuda")
    )
    window = state.prepare_inputs(input_batch, req_states)["lookback_token_ids"]
    assert window.cpu().tolist() == [
        [22, 21, 16],
        [-1, -1, -1],
        [0, 0, 0],
        [-1, -1, -1],
    ]
    # Graph capture runs on the dummy inputs and replays read the same buffer.
    dummy = state.prepare_dummy_inputs(num_reqs=3, num_tokens=3)["lookback_token_ids"]
    assert dummy is window and torch.all(dummy == -1)


def _reference_lookup(weight, scale_inv, ids, start, end, block=32):
    """The torch expression the fused kernel replaces."""
    mask = (ids < start) | (ids >= end)
    local = (ids - start).masked_fill(mask, 0).long()
    values = torch.nn.functional.embedding(local, weight)
    scales = torch.nn.functional.embedding(local, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.float().unflatten(-1, (-1, block))
    values = (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


def _make_embedding(cpu_offload, rows=4096, dim=256, block=32):
    layer = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.dim, layer.block_size, layer.tp_size = dim, block, 1
    layer.n_hash_cols = layer.part_n_hash_cols = 24
    layer.head_start = 0
    layer.cpu_offload = cpu_offload
    layer.part_num_embeddings = rows
    # A window strictly inside the table, so unowned rows are exercised too.
    layer.vocab_start_idx, layer.vocab_end_idx = rows // 4, rows // 4 + rows // 2
    layer._views = layer._view_src = None
    layer._num_sms = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    kwargs = (
        {"device": "cpu", "pin_memory": True} if cpu_offload else {"device": "cuda"}
    )
    owned = layer.vocab_end_idx - layer.vocab_start_idx
    layer.weight = torch.empty(owned, dim, dtype=torch.float8_e4m3fn, **kwargs)
    layer.weight_scale_inv = torch.empty(
        owned, dim // block, dtype=torch.uint8, **kwargs
    )
    torch.manual_seed(0)
    layer.weight.copy_((torch.randn(owned, dim) * 4).to(torch.float8_e4m3fn))
    layer.weight_scale_inv.copy_(
        torch.randint(120, 134, (owned, dim // block), dtype=torch.uint8)
    )
    return layer


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_engram_head_shards_reconstruct_checkpoint(cpu_offload, tp_size, monkeypatch):
    """Keep complete buckets and reconstruct head order, including TP padding."""
    head_sizes = (17, 19, 23, 29, 31, 37)
    num_rows, dim = sum(head_sizes), 64
    torch.manual_seed(0)
    weight = torch.randn(num_rows + 7, dim).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (num_rows + 7, dim // 32), dtype=torch.uint8)
    hashes = torch.empty(7, 2, len(head_sizes), dtype=torch.int32, device="cuda")
    start = 0
    for head, size in enumerate(head_sizes):
        hashes[:, :, head].random_(start, start + size)
        hashes[0, :, head] = start
        hashes[-1, :, head] = start + size - 1
        start += size
    ids = hashes[:, 1]
    expected = _reference_lookup(weight.cuda(), scales.cuda(), ids, 0, num_rows)
    monkeypatch.setattr(
        engram_ops, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    shards = []
    layers = []
    for rank in range(tp_size):
        monkeypatch.setattr(
            engram_ops, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        with torch.device("cuda"):
            layer = ParallelEngramEmbedding(
                num_rows + 7, dim, head_sizes, cpu_offload=cpu_offload
            )
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
        )
        out = torch.empty(
            len(ids),
            layer.part_n_hash_cols,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        layer.lookup(ids, out)
        shards.append(out)
        layers.append(layer)
    gathered = torch.cat(shards, dim=1)
    torch.testing.assert_close(gathered[:, : len(head_sizes)], expected, rtol=0, atol=0)
    assert torch.count_nonzero(gathered[:, len(head_sizes) :]) == 0
    assert sum(layer.part_num_embeddings for layer in layers) == num_rows

    def gather(local, dim):
        torch.testing.assert_close(local, shards[0], rtol=0, atol=0)
        return torch.cat(shards, dim=dim)

    monkeypatch.setattr(engram_ops, "tensor_model_parallel_all_gather", gather)
    torch.testing.assert_close(layers[0](ids), expected, rtol=0, atol=0)
    module = Engram.__new__(Engram)
    torch.nn.Module.__init__(module)
    module.embed_tokens = layers[0]
    module.use_sequence_parallel = False
    module.staged_rows = torch.empty_like(shards[0])
    module.prepare_embeddings(ids)
    torch.testing.assert_close(module.embed(ids), expected, rtol=0, atol=0)
    # Slicing before head reordering must preserve padded heads and empty owners.
    module.use_sequence_parallel = True
    chunk = (len(ids) + tp_size - 1) // tp_size
    padded = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, (-len(ids)) % tp_size))
    for rank in range(tp_size):
        monkeypatch.setattr(
            engram_ops, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        torch.testing.assert_close(
            module.embed(ids), padded[rank * chunk : (rank + 1) * chunk], rtol=0, atol=0
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 256])
def test_engram_lookup_matches_torch(cpu_offload, background, num_tokens):
    """The fused gather must be bit-exact with the torch dequant path it
    replaces, from HBM and from pinned host memory alike, and must contribute
    zeros for rows another TP rank owns."""
    layer = _make_embedding(cpu_offload)
    cols, rows = 24, layer.part_num_embeddings
    ids = torch.randint(0, rows, (num_tokens, cols), dtype=torch.int32, device="cuda")
    expected = _reference_lookup(
        layer.weight.cuda(),
        layer.weight_scale_inv.cuda(),
        ids,
        layer.vocab_start_idx,
        layer.vocab_end_idx,
    )
    out = torch.empty(num_tokens, cols, layer.dim, dtype=torch.bfloat16, device="cuda")
    layer.lookup(ids, out, background=background)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("capture", ["eager", "full", "breakable"])
def test_engram_prepared_rows_survive_graph_breaks(cpu_offload, capture):
    """Consume early lookup results after a break, with fresh IDs each replay."""
    _run_engram_prepared_rows(cpu_offload, capture)


def _run_engram_prepared_rows(
    cpu_offload, capture, tp_size=1, rank=0, use_sequence_parallel=False
):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    layer = _make_embedding(cpu_offload)
    cols, num_tokens = (23, 65) if use_sequence_parallel else (24, 64)
    layer.n_hash_cols = cols
    layer.tp_size = tp_size
    layer.part_n_hash_cols = (cols + tp_size - 1) // tp_size
    layer.head_start = rank * layer.part_n_hash_cols
    engram = Engram.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = layer
    engram.use_sequence_parallel = use_sequence_parallel
    engram.staged_rows = torch.empty(
        num_tokens,
        layer.part_n_hash_cols,
        layer.dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Match the non-contiguous per-layer slice of the model hash tensor.
    hashes = torch.randint(
        0,
        layer.part_num_embeddings,
        (num_tokens, 2, cols),
        dtype=torch.int32,
        device="cuda",
    )
    src = hashes[:, 1]
    local_tokens = (
        (num_tokens + tp_size - 1) // tp_size if use_sequence_parallel else num_tokens
    )
    out = torch.empty(
        local_tokens, cols, layer.dim, dtype=torch.bfloat16, device="cuda"
    )
    embed = engram.embed
    if capture == "compiled":
        embed = torch.compile(embed, backend="eager", fullgraph=True, dynamic=True)

    def step(cap=None):
        engram.prepare_embeddings(src)
        if cap is not None:
            cap.add_eager(lambda: None)
        out.copy_(embed(src))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)

    graph = None
    if capture == "full":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    elif capture == "breakable":
        graph = BreakableCUDAGraphCapture()
        with torch.cuda.stream(warmup), graph:
            step(graph)
        torch.cuda.current_stream().wait_stream(warmup)
        assert graph.num_graphs == 2

    for _ in range(3):
        hashes.random_(0, layer.part_num_embeddings)
        if graph is None:
            step()
        else:
            graph.replay()
        expected = _reference_lookup(
            layer.weight.cuda(),
            layer.weight_scale_inv.cuda(),
            src,
            layer.vocab_start_idx,
            layer.vocab_end_idx,
        )
        if use_sequence_parallel:
            expected = torch.nn.functional.pad(
                expected, (0, 0, 0, 0, 0, (-num_tokens) % tp_size)
            )[rank * local_tokens : (rank + 1) * local_tokens]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _engram_tp_worker(rank, tp_size, port):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import cleanup_dist_env_and_memory

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(tp_size, 1, rank, str(port), local_rank=rank)
    try:
        for cpu_offload in (False, True):
            for sp in (False, True):
                for capture in ("eager", "compiled", "full", "breakable"):
                    _run_engram_prepared_rows(cpu_offload, capture, tp_size, rank, sp)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_engram_head_collectives_survive_graph_breaks():
    """All-gather preserves head order and local SP tokens across graph replay."""
    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two GPUs")
    torch.multiprocessing.spawn(_engram_tp_worker, args=(2, get_open_port()), nprocs=2)
