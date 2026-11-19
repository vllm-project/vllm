# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from tests.utils import _test_rendezvous_path, init_test_distributed_environment
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the Muse fused Markov path is selected on SM100-family devices",
)
@pytest.mark.parametrize("rows", [1, 3, 48])
@torch.inference_mode()
def test_fused_markov_chain_feeds_confidence_head_its_sampled_tokens(rows):
    """Adaptive verification keeps the fused chain: each step's confidence
    reads the Markov embedding of the token that step actually sampled."""
    from types import SimpleNamespace

    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov_chain import (  # noqa: E501
        build_markov_chain,
    )
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    torch.manual_seed(185)
    steps, vocab, rank, hidden_size = 7, 2051, 512, 64
    with torch.device("cuda"):
        w1 = torch.randn(vocab, rank, dtype=torch.bfloat16)
        w2 = torch.randn(vocab, rank, dtype=torch.bfloat16) / rank**0.5
        conf_w = torch.randn(hidden_size + rank)

        def compute_confidence(hidden, markov_embed):
            x = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
            return torch.sigmoid(x @ conf_w)

        draft = SimpleNamespace(
            model=SimpleNamespace(
                markov_head=SimpleNamespace(
                    markov_w1=SimpleNamespace(weight=w1),
                    markov_w2=SimpleNamespace(weight=w2),
                )
            ),
            config=SimpleNamespace(draft_vocab_size=vocab),
            logits_processor=SimpleNamespace(head_dtype=None),
            draft_id_to_target_id=None,
            output_logit_transform_params=lambda: (0.5, 30.0),
            markov_bias=lambda emb: emb @ w2.T,
            compute_confidence=compute_confidence,
        )
        spec = DSparkSpeculator.__new__(DSparkSpeculator)
        spec.model = draft
        spec.num_speculative_steps = steps
        spec.max_num_reqs = rows
        spec.use_confidence_head = True
        spec.use_acceptance_estimator = False
        spec._draft_topk = None
        spec.draft_watermarker = None
        spec.use_fp64_gumbel = False
        spec.sample_from_anchor = True
        spec.draft_logits = DraftLogitsCache(
            rows, steps, vocab, dtype=torch.bfloat16, fill=0.0, device="cuda"
        )
        spec.seeds = torch.arange(rows, dtype=torch.int64)
        spec.temperature = torch.tensor([0.0, 1.0], dtype=torch.float32).repeat(rows)[
            :rows
        ]
        chain = build_markov_chain(draft, spec)
        assert chain is not None
        draft.sample_markov_chain = lambda s, *args, **kwargs: chain.run(
            s, *args, **kwargs
        )

        input_ids = torch.randint(vocab, (rows * 3,), dtype=torch.int32)
        anchors = torch.arange(rows, dtype=torch.int64) * 3 + 1
        base_logits = torch.randn(rows, steps, vocab, dtype=torch.bfloat16)
        hidden = torch.randn(rows * steps, hidden_size, dtype=torch.bfloat16)
        idx_map = torch.arange(rows).unsqueeze(1).repeat(1, steps)
        sample_pos = 100 + torch.arange(steps).expand(rows, -1).contiguous()
        tokens = torch.full((rows, steps), -1, dtype=torch.int64)

        confidence = spec._sample_markov_chain(
            base_logits,
            hidden,
            input_ids,
            idx_map,
            sample_pos,
            tokens,
            prev_indices=anchors,
        )

        assert confidence is not None and confidence.shape == (rows, steps)
        assert torch.all((tokens >= 0) & (tokens < vocab))
        prev = torch.cat([input_ids[anchors].long().unsqueeze(1), tokens], dim=1)
        expected = compute_confidence(hidden, w1[prev[:, :steps]].flatten(0, 1)).view(
            rows, steps
        )
        torch.testing.assert_close(confidence, expected, rtol=0, atol=0)


requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the Muse CuTe draft and sampling kernels run on SM100-family devices",
)

TP, SHARD, STEPS, MARKOV_RANK = 8, 25256, 7, 512
VOCAB = TP * SHARD
MULTIPLIER, CAP = 0.17677669529663687, 20.0
MAX_REQS = 16


def _tp2_worker(rank: int, check, port: str) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        torch.accelerator.set_device_index(rank)
        try:
            init_test_distributed_environment(2, 1, rank, port, local_rank=rank)
            check(2)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def run_tp2(check, timeout: float = 1200.0) -> None:
    """`check(world)` on each rank of a TP2 group, one process per GPU; fails
    (killing the ranks) when they take more than `timeout` seconds."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    from vllm.utils.network_utils import get_open_port

    port = str(get_open_port())
    try:
        context = mp.spawn(_tp2_worker, args=(check, port), nprocs=2, join=False)
        deadline = time.monotonic() + timeout
        while not context.join(max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                for process in context.processes:
                    process.kill()
                pytest.fail(f"the TP2 ranks hung for {timeout} s")
    finally:
        Path(_test_rendezvous_path(port)).unlink(missing_ok=True)


def _step(kernels, base, bias, cache, i, state, tokens, emb, offset, **kwargs):
    idx_map, pos, seeds, temperature, argmax, maximum, arrivals = state
    kernels.step(
        base,
        bias,
        cache,
        i,
        idx_map[:, i],
        pos[:, i],
        seeds,
        temperature,
        argmax,
        maximum,
        arrivals,
        tokens,
        emb,
        offset,
        MULTIPLIER,
        1.0 / CAP,
        CAP,
        **kwargs,
    )


def _chain_inputs(rows, width, world, rank, call, generator):
    """One call's `(base, bias, temperature)` on `rank`: random logits, plus
    rows whose winner is fixed across the shards: a tie between the last
    entry of one shard and the first of the next (the lower id wins), and
    NaN entries on two ranks (the first NaN wins)."""
    base = torch.randn(rows, STEPS, width, generator=generator) * 3
    bias = torch.randn(STEPS, rows, width, generator=generator)
    temperature = torch.tensor([1.0, 0.0]).repeat(rows)[:rows]
    for row in range(1, rows, 3):
        temperature[row] = 0.0
        low = (call + row) % (world - 1)
        base[row].fill_(-30.0)
        bias[:, row].zero_()
        if rank == low:
            base[row, :, width - 1] = 20.0
        if rank == low + 1:
            base[row, :, 0] = 20.0
    for row in range(2, rows, 3):
        if rank in (world - 1, (call + row) % world):
            base[row, :, 7 + rank] = float("nan")
    return base.bfloat16(), bias.bfloat16(), temperature


def _check_chain(world):
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group, graph_capture
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.draft import (
        markov as kernels,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov import (
        STEP_PDL_MAX_ROWS,
        step_block,
    )

    group = get_tp_group()
    rank = group.rank_in_group
    device = torch.device("cuda", torch.accelerator.current_device_index())
    calls, vocab, offset = 6, SHARD * world, rank * SHARD
    generator = torch.Generator().manual_seed(17 + 1000 * rank)
    torch.manual_seed(17)
    w1 = torch.randn(vocab, MARKOV_RANK, dtype=torch.bfloat16).to(device)
    chain = kernels.PacketMailbox(
        group.device_group,
        tp=world,
        rank=rank,
        steps=STEPS,
        rows=8,
        vocab=vocab,
        device=device,
        cache_shard=8 * STEPS * SHARD,
    )
    variants = {(step_block(r), r <= STEP_PDL_MAX_ROWS) for r in range(1, 9)}
    with torch.device(device):
        embeds = torch.empty(calls * (STEPS + 1) * 8 * MARKOV_RANK).bfloat16()
        loop = kernels.MarkovKernels(
            w1,
            embeds,
            width=SHARD,
            has_cap=True,
            steps=[(b, p, True) for b, p in variants],
            fill_vector=True,
            chain=chain,
        )
        reference = kernels.MarkovKernels(
            w1,
            embeds,
            width=vocab,
            has_cap=True,
            steps=[(b, p, v) for b, p in variants for v in (False, True)],
        )
        blocks = -(-vocab // min(b for b, _ in variants))
        argmax = torch.empty(8 * blocks, dtype=torch.int64)
        maximum = torch.empty(8 * blocks)
        arrivals = torch.zeros(2, 8, dtype=torch.int32)

    def full_vocab(shard):
        ranks = torch.empty((world, *shard.shape), dtype=shard.dtype, device=device)
        dist.all_gather_into_tensor(ranks, shard, group=group.device_group)
        return ranks.movedim(0, -2).flatten(-2)

    def check_rows(rows):
        kwargs = dict(block=step_block(rows), pdl=rows <= STEP_PDL_MAX_ROWS)
        inputs = [
            _chain_inputs(rows, SHARD, world, rank, call, generator)
            for call in range(calls)
        ]
        base, bias, temperature = (torch.stack(x).to(device) for x in zip(*inputs))
        full_base, full_bias = full_vocab(base), full_vocab(bias)
        idx_map = torch.arange(rows, device=device).unsqueeze(1).repeat(1, STEPS)
        pos = 100 + torch.arange(STEPS, device=device).expand(rows, -1).contiguous()
        seeds = torch.arange(rows, dtype=torch.int64, device=device) + 77
        emb = embeds[: calls * (STEPS + 1) * rows * MARKOV_RANK].view(
            calls, STEPS + 1, rows, MARKOV_RANK
        )
        emb[:, 0].copy_(w1[: calls * rows].view(calls, rows, MARKOV_RANK))
        slot = chain.cache[0, rank, : rows * STEPS * SHARD].view(rows, STEPS, SHARD)
        fill_map = torch.arange(rows * STEPS, device=device) // STEPS
        shape = (calls, rows, STEPS)

        def state(call, which):
            return (
                idx_map,
                pos,
                seeds,
                temperature[call],
                argmax,
                maximum,
                arrivals[which, :rows],
            )

        expected = torch.full(shape, -1, dtype=torch.int64, device=device)
        expected_full = torch.full((*shape, vocab), -1.0, device=device).bfloat16()
        for call in range(calls):
            for i in range(STEPS):
                _step(
                    reference,
                    full_base[call, :, i],
                    full_bias[call, i],
                    expected_full[call],
                    i,
                    state(call, 0),
                    expected[call],
                    emb[call, i + 1],
                    0,
                    **kwargs,
                )
        expected_emb = emb.clone()
        got = torch.full_like(expected, -1)
        full = torch.full_like(expected_full, -1.0)

        def run(call, sleep=False):
            if sleep and rank == call % world:
                torch.cuda._sleep(200_000)
            for i in range(STEPS):
                _step(
                    loop,
                    base[call, :, i],
                    bias[call, i],
                    slot,
                    i,
                    state(call, 1),
                    got[call],
                    emb[call, i + 1],
                    offset,
                    **kwargs,
                )
            if sleep and rank == (call + 1) % world:
                torch.cuda._sleep(200_000)
            kernels.cache_fill(chain, full[call], fill_map, rows, STEPS, SHARD, vocab)

        def check():
            torch.accelerator.synchronize()
            assert torch.equal(got, expected), (rows, rank)
            assert torch.equal(emb, expected_emb)
            assert torch.equal(full.view(torch.int16), expected_full.view(torch.int16))

        for call in range(calls):
            run(call)
            torch.accelerator.synchronize()
            dist.barrier(group=group.cpu_group)
        check()

        graph = torch.cuda.CUDAGraph()
        with graph_capture(device) as context:
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph, stream=context.stream):
                for call in range(calls):
                    run(call, sleep=True)
        for _ in range(2):
            got.fill_(-1)
            emb[:, 1:].zero_()
            full.fill_(-1.0)
            graph.replay()
            check()
        dist.barrier(group=group.cpu_group)

    with pytest.raises(ValueError, match="steps"):
        kernels.cache_fill(
            chain,
            torch.empty(1, STEPS, vocab, dtype=torch.bfloat16, device=device),
            torch.zeros(STEPS - 1, dtype=torch.int64, device=device),
            1,
            STEPS - 1,
            SHARD,
            vocab,
        )
    for rows in range(1, 9):
        check_rows(rows)


@requires_sm100
def test_chained_markov_step_samples_as_the_unsharded_step():
    """The chained step (`PacketMailbox`) samples, embeds and fills the cache
    as the unsharded step over the full vocabulary, across shard-boundary ties
    and NaN rows, eagerly and in CUDA graph replays with skewed ranks."""
    run_tp2(_check_chain)


def _check_multicast_agreement(world):
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.common import (
        mailbox,
    )

    group = get_tp_group()
    rank = group.rank_in_group
    rendezvous = symm_mem.rendezvous

    def without_multicast(tensor, group):
        handle = rendezvous(tensor, group)
        return SimpleNamespace(buffer_ptrs=handle.buffer_ptrs, multicast_ptr=0)

    if rank == 1:
        symm_mem.rendezvous = without_multicast
    try:
        with pytest.raises(RuntimeError, match="multicast"):
            mailbox.LamportMailbox(
                group.device_group,
                tp=world,
                rank=rank,
                capacity=1,
                width=8,
                device=torch.device("cuda", rank),
            )
    finally:
        symm_mem.rendezvous = rendezvous


@requires_sm100
@pytest.mark.distributed(num_gpus=2)
def test_mailbox_without_multicast_on_one_rank_raises_on_every_rank():
    """A multicast binding that fails on one rank only makes every rank's
    mailbox construction raise, instead of the others hanging in the next
    collective."""
    run_tp2(_check_multicast_agreement, timeout=180.0)


def test_vocab_sharded_verification_falls_back_without_multicast(
    monkeypatch, caplog_vllm
):
    """A candidate mailbox without a multicast mapping (on every rank, as
    `_rendezvous` agrees) disables the vocab-sharded verification, so serving
    takes the stock sampler, instead of failing the engine start-up."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.common.symmetric import (  # noqa: E501
        SymmetricBuffer,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample import (
        shard_top_k,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.sample import (
        vocab_sharded,
    )

    def no_multicast(group, shape, dtype, device):
        return SymmetricBuffer(torch.zeros(shape, dtype=dtype), (), 0)

    monkeypatch.setattr(shard_top_k, "symmetric_buffer", no_multicast)
    monkeypatch.setattr(vocab_sharded, "_unsupported_reason", lambda *_: None)
    monkeypatch.setattr(
        vocab_sharded,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=TP, device_group="tp"),
    )
    build = vocab_sharded.VocabShardedVerifier.build
    assert build(None, None, None, torch.device("cpu")) is None
    assert "Vocab-sharded verification disabled" in caplog_vllm.text


def _alias(address: int, numel: int, device: torch.device) -> torch.Tensor:
    """An int32 tensor on `device` over `numel` elements at `address` (a peer
    or multicast mapping is accessible from this device only)."""
    storage = torch._C._construct_storage_from_data_pointer(address, device, numel * 4)
    return torch.empty(0, dtype=torch.int32, device=device).set_(storage, 0, (numel,))


def _check_arena(world):
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.common import (
        symmetric,
    )

    group = get_tp_group()
    rank = group.rank_in_group
    device = torch.device("cuda", rank)
    arena = symmetric.MulticastArena(group.device_group, device, chunk_bytes=4096)
    # (shape, dtype) -> (chunk, offset): the second straddles the first
    # chunk's end, the third exceeds chunk_bytes, the fourth fills a fresh
    # chunk and the fifth follows it at the next ARENA_ALIGN boundary.
    layout = [
        ((100,), torch.uint8, 0, 0),
        ((1000,), torch.int32, 1, 0),
        ((3, 4096), torch.uint8, 2, 0),
        ((5,), torch.int32, 3, 0),
        ((7,), torch.int64, 3, symmetric.ARENA_ALIGN),
    ]
    buffers = [arena.allocate(shape, dtype) for shape, dtype, _, _ in layout]
    chunks = arena.chunks
    assert [c.numel() for c in chunks] == [4096, 4096, 3 * 4096, 4096]
    for buffer, (shape, dtype, chunk, offset) in zip(buffers, layout):
        assert buffer.tensor.shape == shape and buffer.tensor.dtype == dtype
        assert buffer.tensor.data_ptr() == chunks[chunk].data_ptr() + offset
        assert buffer.peers[rank] == buffer.tensor.data_ptr()
        assert buffer.multicast

    def words(buffer):
        return buffer.tensor.view(-1).view(torch.int32)

    def exchange(store):
        """Rank 0 stores i + 1 into buffer i at `store(buffer)`; every rank's
        buffers after it."""
        for buffer in buffers:
            words(buffer).zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=group.cpu_group)
        if rank == 0:
            for i, buffer in enumerate(buffers):
                _alias(store(buffer), words(buffer).numel(), device).fill_(i + 1)
        torch.accelerator.synchronize()
        dist.barrier(group=group.cpu_group)
        return [words(buffer).tolist() for buffer in buffers]

    # Rank 0 stores into rank 1's buffers through its peer addresses, then
    # into every rank's through the multicast addresses.
    peer = exchange(lambda buffer: buffer.peers[1])
    multicast = exchange(lambda buffer: buffer.multicast)
    for i, buffer in enumerate(buffers):
        assert peer[i] == [(i + 1) * rank] * words(buffer).numel(), (i, rank)
        assert multicast[i] == [i + 1] * words(buffer).numel(), (i, rank)


@requires_sm100
@pytest.mark.distributed(num_gpus=2)
def test_multicast_arena_places_buffers_in_chunks():
    """`MulticastArena.allocate` rolls a buffer that does not fit over to a new
    chunk, gives one larger than a chunk its own, aligns offsets to
    `ARENA_ALIGN`, and offsets its peer and multicast addresses to match."""
    run_tp2(_check_arena, timeout=180.0)


def test_multicast_arena_refuses_to_nest():
    """An arena opened inside an active one for the same group raises, rather
    than silently carving from the enclosing arena's chunks."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.common import (
        symmetric,
    )

    device = torch.device("cpu")
    with symmetric.multicast_arena("tp", device):
        with (
            pytest.raises(RuntimeError, match="already"),
            symmetric.multicast_arena("tp", device, chunk_bytes=4096),
        ):
            pass
        with symmetric.multicast_arena("other", device):
            pass


@requires_sm100
@torch.inference_mode()
def test_markov_bias_table_samples_as_the_gemm_loop():
    """Each step reading its bias from `build_bias_table`'s table at the rows'
    tokens samples, caches and embeds bitwise as the per-step w2 GEMM, at 1-8
    rows of the served vocab shard."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.draft import (
        markov as kernels,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov import (
        STEP_PDL_MAX_ROWS,
        VocabParallelMarkovHead,
        build_bias_table,
        step_block,
    )

    torch.manual_seed(892)
    with torch.device("cuda"):
        w1 = torch.randn(SHARD + 100, MARKOV_RANK).bfloat16()
        w2 = (torch.randn(SHARD, MARKOV_RANK) / MARKOV_RANK**0.5).bfloat16()
    head = SimpleNamespace(markov_w2=SimpleNamespace(weight=w2), _local_w2=w2)
    head.local_bias = functools.partial(VocabParallelMarkovHead.local_bias, head)
    table = build_bias_table(head, w1)
    variants = {(step_block(r), r <= STEP_PDL_MAX_ROWS) for r in range(1, 9)}
    with torch.device("cuda"):
        embeds = torch.empty((STEPS + 1) * 8 * MARKOV_RANK).bfloat16()
        loops = [
            kernels.MarkovKernels(
                w1,
                embeds,
                width=SHARD,
                has_cap=True,
                steps=[(b, p, v) for b, p in variants for v in (False, True)],
                table=table_mode,
            )
            for table_mode in (False, True)
        ]
        blocks = -(-SHARD // min(b for b, _ in variants))
        argmax = torch.empty(8 * blocks, dtype=torch.int64)
        maximum = torch.empty(8 * blocks)
        arrivals = torch.zeros(8, dtype=torch.int32)
        anchor_tokens = torch.empty(8, dtype=torch.int64)

    for rows in range(1, 9):
        kwargs = dict(block=step_block(rows), pdl=rows <= STEP_PDL_MAX_ROWS)
        with torch.device("cuda"):
            base = (torch.randn(rows, STEPS, SHARD) * 3).bfloat16()
            input_ids = torch.randint(w1.shape[0], (rows * 2,), dtype=torch.int32)
            # An id past the table reads its last row, as its w1 row copy does.
            input_ids[0] = w1.shape[0] + 5
            anchors = torch.arange(rows) * 2
            state = (
                torch.arange(rows).unsqueeze(1).repeat(1, STEPS),
                100 + torch.arange(STEPS).expand(rows, -1).contiguous(),
                torch.arange(rows, dtype=torch.int64) + 11,
                torch.tensor([1.0, 0.0]).repeat(rows)[:rows],
                argmax,
                maximum,
                arrivals[:rows],
            )
        emb = embeds[: (STEPS + 1) * rows * MARKOV_RANK].view(
            STEPS + 1, rows, MARKOV_RANK
        )
        outputs = []
        for table_mode, loop in enumerate(loops):
            tokens = torch.full((rows, STEPS), -1, dtype=torch.int64, device="cuda")
            cache = torch.full((rows, STEPS, SHARD), -1.0, device="cuda").bfloat16()
            anchor = anchor_tokens[:rows] if table_mode else None
            kernels.anchor_embed(
                input_ids, anchors, w1, emb[0], arrivals[:rows], anchor
            )
            for i in range(STEPS):
                if table_mode:
                    bias, token = table, anchor if i == 0 else tokens[:, i - 1]
                else:
                    bias, token = head.local_bias(emb[i]), None
                _step(
                    loop,
                    base[:, i],
                    bias,
                    cache,
                    i,
                    state,
                    tokens,
                    emb[i + 1],
                    0,
                    bias_token=token,
                    **kwargs,
                )
            outputs.append((tokens, emb.clone(), cache.view(torch.int16)))
        for got, expected in zip(*outputs):
            assert torch.equal(got, expected), rows


def _dequant_nvfp4(codes, scales, tensor_scale):
    from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype

    return dequantize_nvfp4_to_dtype(
        codes,
        scales,
        1.0 / tensor_scale.float(),
        torch.float32,
        codes.device,
        is_sf_linear_layout=True,
    )


@requires_sm100
@pytest.mark.parametrize("x_gain", [1.0, 3e4])
@torch.inference_mode()
def test_draft_w4a16_mlp_matches_fp32_reference(x_gain):
    """`DraftW4A16Mlp` at rows 1-8 against the FP32 arithmetic of the same
    NVFP4 weights with x and the SiLU product rounded to F16 at their
    power-of-two scales (`x_gain` 3e4 takes an input scale below one)."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.draft.fp4_mlp import (  # noqa: E501
        DraftW4A16Mlp,
        power_of_two_scale,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.quantize_experts import (  # noqa: E501
        quantize_nvfp4_weight,
    )

    hidden, inter = 8192, 2048
    torch.manual_seed(0)
    with torch.device("cuda"):
        gate_up = (torch.randn(2 * inter, hidden) * 0.02).bfloat16()
        down = (torch.randn(hidden, inter) * 0.02).bfloat16()
        x8 = (torch.randn(8, hidden) * x_gain).bfloat16()

    def weight(w):
        return _dequant_nvfp4(*quantize_nvfp4_weight(w.contiguous()))

    gate, up = weight(gate_up[:inter]), weight(gate_up[inter:])
    w_down = weight(down)
    silu, xf8, g = torch.nn.functional.silu, x8.float(), gate_up.float()
    inter_amax = (silu(xf8 @ g[:inter].T) * (xf8 @ g[inter:].T)).abs().max().item()
    input_amax = xf8.abs().max().item()
    mlp = DraftW4A16Mlp(gate_up, down, input_amax, inter_amax, 8)
    x_scale, inter_scale = (
        power_of_two_scale(input_amax),
        power_of_two_scale(inter_amax),
    )
    for m in range(1, 9):
        x = x8[:m].contiguous()
        xf = (x.float() * x_scale).half().float() / x_scale
        h = torch.nn.functional.silu(xf @ gate.T) * (xf @ up.T)
        expected = ((h * inter_scale).half().float() / inter_scale) @ w_down.T
        got = mlp.run(x).float()
        assert ((got - expected).norm() / expected.norm()).item() < 5e-3, m


def test_draft_mlp_nvfp4_scales_refuses_a_layer_without_a_dense_mlp():
    """`draft_mlp_nvfp4` running `scales` raises its documented ValueError,
    not an AttributeError, for a draft layer without a dense MLP."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.draft import (
        _fp4_mlps,
    )

    model = SimpleNamespace(layers=[SimpleNamespace(feed_forward=SimpleNamespace())])
    setup = SimpleNamespace(amax=[(1.0, 1.0)], option=SimpleNamespace(prefix="x"))
    all_reduce = SimpleNamespace(mailbox=SimpleNamespace(capacity=8))
    with pytest.raises(ValueError, match="draft_mlp_nvfp4: layer 0"):
        _fp4_mlps(model, setup, all_reduce)


@requires_sm100
@pytest.mark.parametrize("rows", range(1, 9))
def test_nvfp4_lm_head_tracks_the_bf16_head(rows):
    """The W4A4 head's logits are within the E2M1 error of the BF16 head's
    (relative L2 about 0.13 here) and keep most of each row's top 64. Without
    the specialized decode (`--monokernel`) the option is refused, and so is a
    misspelled one."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import muse_options
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.nvfp4_lm_head import (  # noqa: E501
        Nvfp4LMHeadMethod,
    )

    with pytest.raises(ValueError, match="decode_backend"):
        muse_options(SimpleNamespace(additional_config={"nvfp4_lm_head": True}))
    monokernel = SimpleNamespace(
        additional_config={"nvfp4_lm_head": True},
        kernel_config=SimpleNamespace(monokernel=True),
    )
    assert muse_options(monokernel).nvfp4_lm_head
    with pytest.raises(ValueError, match="'nvfp4_lmhead'"):
        muse_options(SimpleNamespace(additional_config={"nvfp4_lmhead": True}))
    assert (
        muse_options(
            SimpleNamespace(additional_config={"kda_decode_backend": "auto"})
        ).decode_backend
        == "auto"
    )
    torch.manual_seed(rows)
    head = torch.nn.Linear(8192, SHARD, bias=False, device="cuda", dtype=torch.bfloat16)
    head.requires_grad_(False).weight.normal_(0, 0.05)
    method = Nvfp4LMHeadMethod()
    method.process_weights_after_loading(head)
    x = torch.randn(rows, 8192, device="cuda").bfloat16()
    got = method.apply(head, x).float()
    expected = x.float() @ head.weight.float().t()
    assert got.shape == (rows, SHARD)
    assert (got - expected).norm() / expected.norm() < 0.2
    top = expected.topk(64, dim=-1).indices
    hits = [
        torch.isin(got[i].topk(64).indices, top[i]).float().mean() for i in range(rows)
    ]
    assert torch.stack(hits).mean() > 0.5


def _soft_capped(raw: torch.Tensor) -> torch.Tensor:
    from vllm import _custom_ops as ops

    return ops.muse_spark_mul_tanh_soft_cap(raw, MULTIPLIER, CAP)


@functools.cache
def _cached(cls, *args):
    return cls(*args)


def _mailbox(tp: int, rows: int):
    """A local `CandidateMailbox` that one process posts every rank's packets
    to."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample.shard_top_k import (  # noqa: E501
        CandidateMailbox,
    )

    return _cached(CandidateMailbox, tp, rows, "cuda")


def _post_shards(raw: torch.Tensor, mailbox) -> None:
    """Post the TP8 shards' `ShardTopK` packets to `mailbox`."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample.shard_top_k import (  # noqa: E501
        ShardTopK,
    )

    top_k = _cached(ShardTopK, SHARD, 64, raw.device, False)
    for rank in range(TP):
        shard = raw[:, rank * SHARD : (rank + 1) * SHARD]
        top_k.run(shard, MULTIPLIER, CAP, rank * SHARD, mailbox, rank)


def _post(packets: torch.Tensor, mailbox, seqs: torch.Tensor | None = None) -> None:
    """Post `[tp, rows, PACKET_WORDS]` (key, column) packets to `mailbox` as
    `ShardTopK` does: each word tagged with its row's sequence number (or
    `seqs`), in that number's generation."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample.shard_top_k import (  # noqa: E501
        TAG_BITS,
    )

    rows = packets.shape[1]
    seqs = (mailbox.seqs[:rows] if seqs is None else seqs).long()
    tags = seqs[None, :, None]
    keys = packets[..., 0::2].long() & 0xFFFFFFFF
    columns = packets[..., 1::2].long()
    high = keys & ~((1 << TAG_BITS) - 1) | tags
    low = ((keys << (32 - TAG_BITS)) | (columns << TAG_BITS) | tags) & 0xFFFFFFFF
    words = torch.stack([high, low], dim=-1).flatten(-2)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    row = torch.arange(rows, device=packets.device)
    mailbox.tensor[seqs & 1, :, row] = words.transpose(0, 1)


def _verify_case(num_reqs: int, seed: int):
    """Greedy requests at the TP8 vocab, the first still in its prefill
    (placeholder drafts)."""
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = num_reqs * (STEPS + 1)
    raw = 60 * torch.randn(rows, VOCAB, device="cuda", generator=generator)
    raw = raw.bfloat16()
    logits = _soft_capped(raw)
    states = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(seed))
    idx_mapping = states[:num_reqs].cuda()
    draft = DraftLogitsCache(
        MAX_REQS, STEPS, VOCAB, dtype=torch.bfloat16, fill=0.0, device="cuda"
    )
    target = torch.roll(logits.argmax(dim=1), 1)
    other = torch.randint(0, VOCAB, (rows,), device="cuda", generator=generator)
    likely = torch.rand(rows, device="cuda", generator=generator) < 0.7
    prefill_len = torch.full((MAX_REQS,), 10, dtype=torch.int32, device="cuda")
    prefill_len[idx_mapping[0]] = 2000
    return SimpleNamespace(
        rows=rows,
        logits=logits,
        raw=raw,
        draft=draft,
        cu=torch.arange(0, rows + 1, STEPS + 1, dtype=torch.int32, device="cuda"),
        idx_mapping=idx_mapping,
        expanded=idx_mapping.repeat_interleave(STEPS + 1),
        local=torch.arange(STEPS + 1, dtype=torch.int32, device="cuda").repeat(
            num_reqs
        ),
        input_ids=torch.where(likely, target, other).to(torch.int32),
        positions=1000 + torch.arange(rows, device="cuda") * 3,
        logits_indices=torch.arange(rows, device="cuda"),
        prefill_len=prefill_len,
        seq_lens=torch.full((num_reqs,), 5000, dtype=torch.int32, device="cuda"),
        seeds=torch.randint(
            -(2**62), 2**62, (MAX_REQS,), device="cuda", generator=generator
        ),
        temperature=torch.zeros(MAX_REQS, device="cuda"),
        top_k=torch.full((MAX_REQS,), 64, dtype=torch.int32, device="cuda"),
        top_p=torch.ones(MAX_REQS, device="cuda"),
        min_p=torch.zeros(MAX_REQS, device="cuda"),
    )


def _run_sharded(verify, case, mailbox, rates):
    verify.draft_stats(case.draft.values, case.expanded, case.local, case.temperature)
    verify.select(
        mailbox,
        case.rows,
        case.expanded,
        case.temperature,
        case.top_k,
        case.top_p,
        case.min_p,
    )
    return verify.verify(
        case.draft.values,
        case.cu,
        case.idx_mapping,
        case.logits_indices,
        case.positions,
        case.input_ids,
        case.local,
        case.prefill_len,
        case.seq_lens,
        case.temperature,
        case.seeds,
        rates,
    )


def _stock_verify(case, block: bool, rates):
    """The stock sampler's masks (`Sampler.apply_sampling_params`) and
    `rejection_sample`."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
        gather_draft_sampled,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    draft_sampled, pos = gather_draft_sampled(
        case.input_ids,
        case.positions,
        case.logits_indices,
        case.expanded,
        case.local,
        case.prefill_len,
    )
    logits = case.logits.to(torch.float32, copy=True)
    apply_temperature(logits, case.expanded, case.temperature)
    apply_min_p(logits, case.expanded, case.min_p)
    top_p = case.top_p[case.expanded]
    logits = apply_top_k_top_p(
        logits,
        case.top_k[case.expanded],
        top_p if bool((top_p < 1).any()) else None,
    )
    sampled, num_sampled = rejection_sample(
        logits,
        case.draft,
        draft_sampled,
        case.cu,
        pos,
        case.idx_mapping,
        case.expanded,
        case.local,
        case.temperature,
        case.seeds,
        int(case.local.max()),
        rates,
        use_block_verification=block,
    )
    return sampled, *get_num_sampled_and_rejected(
        num_sampled, case.seq_lens, case.cu, case.idx_mapping, case.prefill_len
    )


def _rates(mode: str, steps: int):
    if mode != "SYNTHETIC":
        return None
    return torch.linspace(0.9, 0.3, steps, device="cuda")


@requires_sm100
@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["STANDARD", "SYNTHETIC", "BLOCK"])
def test_vocab_sharded_verify_greedy_matches_stock(mode, num_reqs):
    """Greedy requests verified from the TP8 shards' `ShardTopK` packets get
    the stock tokens and counts, with a request still in its prefill."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample import (
        verify as sharded,
    )

    verify = _cached(
        sharded.ShardedVerify, TP, VOCAB, 64, STEPS, getattr(sharded, mode), "cuda"
    )
    rates = _rates(mode, STEPS)
    mailbox = _mailbox(TP, 64)
    for trial in range(4):
        case = _verify_case(num_reqs, 100 * num_reqs + trial)
        _post_shards(case.raw, mailbox)
        got = _run_sharded(verify, case, mailbox, rates)
        sampled, num_sampled, num_rejected = _stock_verify(case, mode == "BLOCK", rates)
        assert torch.equal(got[1], num_sampled)
        assert torch.equal(got[2], num_rejected)
        for q, count in enumerate(num_sampled.tolist()):
            assert torch.equal(got[0][q, :count], sampled[q, :count])


# (temperature, top-k, top-p, min-p) of the sampled settings; "mixed" cycles
# through them and greedy.
SAMPLING = {
    "t1_k64": [(1.0, 64, 1.0, 0.0)],
    "t0.7_k20_p0.9": [(0.7, 20, 0.9, 0.0)],
    "t1.3_k50_minp0.05": [(1.3, 50, 1.0, 0.05)],
    "mixed": [
        (1.0, 64, 1.0, 0.0),
        (0.7, 20, 0.9, 0.0),
        (1.3, 50, 1.0, 0.05),
        (0.0, 64, 1.0, 0.0),
    ],
}
SMALL_TP, SMALL_VOCAB, SMALL_STEPS, REPLICAS = 2, 128, 3, 8192


def _sampling_case(settings):
    """`REPLICAS` requests on 128 columns (two 64-column shards, so the
    packets are all the columns), request q with `settings[q % len]` and the
    logits and drafts of its setting."""
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

    generator = torch.Generator(device="cuda").manual_seed(0)
    width = SMALL_STEPS + 1
    kinds = len(settings)
    base = 3 * torch.randn(
        kinds, width, SMALL_VOCAB, device="cuda", generator=generator
    )
    noise = torch.randn(base.shape, device="cuda", generator=generator)
    # Draft tokens: each row's most likely token or a random one.
    tokens = torch.where(
        torch.rand(kinds, width, device="cuda", generator=generator) < 0.6,
        base.argmax(-1),
        torch.randint(SMALL_VOCAB, (kinds, width), device="cuda", generator=generator),
    )
    kind = torch.arange(REPLICAS, device="cuda") % kinds
    rows = REPLICAS * width
    draft = DraftLogitsCache(
        REPLICAS,
        SMALL_STEPS,
        SMALL_VOCAB,
        dtype=torch.bfloat16,
        fill=0.0,
        device="cuda",
    )
    draft.values.copy_((base + noise)[kind, :SMALL_STEPS].bfloat16())
    # The input ids: row r's token is the draft verified at row r - 1.
    input_ids = torch.roll(tokens, 1, dims=1)[kind].flatten().to(torch.int32)
    params = torch.tensor(settings, device="cuda")[kind]
    idx_mapping = torch.arange(REPLICAS, device="cuda")
    return SimpleNamespace(
        kind=kind,
        rows=rows,
        logits=base[kind].flatten(0, 1),
        draft=draft,
        cu=torch.arange(0, rows + 1, width, dtype=torch.int32, device="cuda"),
        idx_mapping=idx_mapping,
        expanded=idx_mapping.repeat_interleave(width),
        local=torch.arange(width, dtype=torch.int32, device="cuda").repeat(REPLICAS),
        input_ids=input_ids,
        positions=1000 + torch.arange(rows, device="cuda"),
        logits_indices=torch.arange(rows, device="cuda"),
        prefill_len=torch.full((REPLICAS,), 10, dtype=torch.int32, device="cuda"),
        seq_lens=torch.full((REPLICAS,), 5000, dtype=torch.int32, device="cuda"),
        seeds=torch.randint(
            -(2**62), 2**62, (REPLICAS,), device="cuda", generator=generator
        ),
        temperature=params[:, 0].contiguous(),
        top_k=params[:, 1].to(torch.int32),
        top_p=params[:, 2].contiguous(),
        min_p=params[:, 3].contiguous(),
    )


def _all_column_packets(logits: torch.Tensor) -> torch.Tensor:
    """`[SMALL_TP, rows, PACKET_WORDS]` packets of every column (in column
    order, `order_key` keys), each rank's 64."""
    bits = logits.contiguous().view(torch.int32)
    keys = torch.where(bits < 0, ~bits, bits | -(2**31))
    columns = torch.arange(SMALL_VOCAB, dtype=torch.int32, device=logits.device)
    words = torch.stack([keys, columns.expand_as(keys)], dim=-1)
    return words.view(-1, SMALL_TP, 64 * 2).transpose(0, 1)


def _kept_columns(row: torch.Tensor, temperature, top_k, top_p, min_p) -> list:
    """The columns a row keeps (as the stock sampler masks it, ties to the
    smaller column), largest first; a greedy row's 64 largest."""
    if temperature == 0.0:
        return torch.sort(row, descending=True, stable=True).indices[:64].tolist()
    scaled = row * (1 / torch.tensor(temperature, dtype=torch.float32))
    order = torch.sort(scaled, descending=True, stable=True).indices
    values = scaled[order]
    count = min(top_k, int((values >= values[0] + math.log(min_p or 1e-45)).sum()))
    if top_p < 1.0:
        probs = torch.exp(values[:count] - values[0])
        before = torch.cumsum(probs, 0) - probs
        count = int((before < top_p * probs.sum()).sum())
    return order[:count].tolist()


def _close(a: torch.Tensor, b: torch.Tensor, n: int) -> bool:
    """Whether two empirical distributions of `n` samples each agree within 6
    standard errors per category (chance failure below 1e-8 per category)."""
    pa, pb = a / n, b / n
    sigma = torch.sqrt((pa * (1 - pa) + pb * (1 - pb)) / n)
    return bool(((pa - pb).abs() <= 6 * sigma + 2 / n).all())


@requires_sm100
@pytest.mark.parametrize("setting", list(SAMPLING))
@pytest.mark.parametrize("mode", ["STANDARD", "SYNTHETIC", "BLOCK"])
def test_vocab_sharded_verify_samples_as_stock(mode, setting):
    """Sampled requests keep the columns the stock sampler's masks keep, and
    their accepted counts and last tokens (the recovered or bonus token) have
    the distribution of the stock `rejection_sample`'s, over many seeds."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample import (
        verify as sharded,
    )

    settings = SAMPLING[setting]
    case = _sampling_case(settings)
    verify = _cached(
        sharded.ShardedVerify,
        SMALL_TP,
        SMALL_VOCAB,
        case.rows,
        SMALL_STEPS,
        getattr(sharded, mode),
        "cuda",
    )
    rates = _rates(mode, SMALL_STEPS)
    mailbox = _mailbox(SMALL_TP, case.rows)
    _post(_all_column_packets(case.logits), mailbox)
    got = _run_sharded(verify, case, mailbox, rates)
    expected = _stock_verify(case, mode == "BLOCK", rates)

    lanes = verify.lanes[: len(settings) * (SMALL_STEPS + 1)].cpu()
    logits = case.logits.cpu()
    for r, lane in enumerate(lanes):
        values, columns = lane[0].view(torch.float32), lane[1]
        kept = columns[values > float("-inf")].tolist()
        reference = _kept_columns(logits[r], *settings[r // (SMALL_STEPS + 1)])
        assert sorted(kept) == sorted(reference), r
        assert kept == sorted(kept), "kept columns in column order"

    for k in range(len(settings)):
        mine = case.kind == k
        n = int(mine.sum())
        counts = []
        for sampled, num_sampled, _ in (got, expected):
            last = sampled[mine].gather(1, num_sampled[mine, None].long() - 1)
            counts.append(
                (
                    torch.bincount(num_sampled[mine].long(), minlength=SMALL_STEPS + 2),
                    torch.bincount(last.flatten(), minlength=SMALL_VOCAB),
                )
            )
        assert _close(counts[0][0].float(), counts[1][0].float(), n), settings[k]
        assert _close(counts[0][1].float(), counts[1][1].float(), n), settings[k]


class _WiringModel:
    """What the model state and the verifier use of Muse Spark's model: a TP
    rank's BF16 head shard and the soft-capped gathered logits."""

    specialized_decode = None

    def __init__(self, rank: int, weight: torch.Tensor):
        self.lm_head = SimpleNamespace(
            weight=weight[rank * SHARD : (rank + 1) * SHARD],
            shard_indices=SimpleNamespace(
                num_org_elements=SHARD,
                num_org_elements_padded=SHARD,
                num_added_elements=0,
                org_vocab_start_index=rank * SHARD,
            ),
        )
        self.specialized_decode_config = SimpleNamespace()
        self.logits_processor = SimpleNamespace(
            logits_as_input=False, soft_cap=None, scale=1.0
        )
        self.mul_tanh_soft_cap = SimpleNamespace(
            multiplier=MULTIPLIER, soft_cap_temp=CAP
        )

    def compute_head_logits_local(self, hidden_states):
        return torch.nn.functional.linear(hidden_states, self.lm_head.weight)

    def compute_logits(self, hidden_states):
        from vllm.distributed import tensor_model_parallel_all_gather

        local = self.compute_head_logits_local(hidden_states).contiguous()
        return _soft_capped(tensor_model_parallel_all_gather(local, -1))


def _wiring_batch(num_reqs: int, generator, logits):
    rows = num_reqs * (STEPS + 1)
    idx_mapping = torch.arange(num_reqs, device="cuda")
    cu = torch.arange(num_reqs + 1, dtype=torch.int32) * (STEPS + 1)
    target = torch.roll(logits.argmax(dim=1), 1)
    other = torch.randint(logits.shape[1], (rows,), device="cuda", generator=generator)
    likely = torch.rand(rows, device="cuda", generator=generator) < 0.7
    return SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping.cpu().numpy(),
        expanded_idx_mapping=idx_mapping.repeat_interleave(STEPS + 1),
        expanded_local_pos=torch.arange(
            STEPS + 1, dtype=torch.int32, device="cuda"
        ).repeat(num_reqs),
        num_draft_tokens=num_reqs * STEPS,
        has_structured_output_reqs=False,
        num_admitted_draft_tokens=None,
        pcp=None,
        logits_indices=torch.arange(rows, device="cuda"),
        cu_num_logits=cu.cuda(),
        cu_num_logits_np=cu.numpy(),
        input_ids=torch.where(likely, target, other).to(torch.int32),
        positions=1000 + torch.arange(rows, device="cuda") * 3,
        seq_lens=torch.full((num_reqs,), 5000, dtype=torch.int32, device="cuda"),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 5000, dtype=torch.int32),
    )


def _check_wiring(world: int) -> None:
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.muse_spark.nvidia.model_state import MuseSparkModelState
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.states import RequestState

    device, vocab, hidden_size = torch.device("cuda"), world * SHARD, 256
    generator = torch.Generator(device=device).manual_seed(0)
    weight = torch.randn(vocab, hidden_size, device=device, generator=generator) / 2
    model = _WiringModel(get_tensor_model_parallel_rank(), weight.bfloat16())
    spec = SimpleNamespace(
        num_speculative_tokens=STEPS,
        enable_adaptive_verification=False,
        rejection_sample_method="standard",
        fly_entropy_threshold=0.3,
        fly_entropy_top_k=3,
        fly_window_size=None,
        draft_sample_method="probabilistic",
    )
    req_states = RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=8192,
        max_num_batched_tokens=64,
        num_speculative_steps=STEPS,
        vocab_size=vocab,
        device=device,
    )
    sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=MAX_REQS,
        vocab_size=vocab,
        device=device,
        req_states=req_states,
        num_speculative_tokens=STEPS + 1,
    )
    state = MuseSparkModelState.__new__(MuseSparkModelState)
    state.vllm_config = SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(
            get_vocab_size=lambda: vocab, head_dtype=torch.bfloat16
        ),
        max_concurrent_batches=3,
    )
    state.model = state.language_model = model
    state.device, state.verifier = device, None
    state.rope_state = state.mm_prefix_ranges = state.prompt_embeds_state = None
    model.specialized_decode_config = None
    assert state.custom_sampler(sampler) is None, "the stock backend samples stock"
    model.specialized_decode_config = SimpleNamespace()
    custom = state.custom_sampler(sampler)
    assert custom is not None, "TP2 with a 25256-column shard verifies vocab-sharded"
    rejection_sampler = custom[1]
    stock = RejectionSampler(sampler, spec, device)
    draft = DraftLogitsCache(
        MAX_REQS, STEPS, vocab, dtype=torch.bfloat16, fill=0.0, device=device
    )
    req_states.prefill_len.np[:] = 10
    req_states.prefill_len.copy_to_uva()
    # The verify outputs of the batches in flight (`max_concurrent_batches`).
    outputs = []
    # Each batch's first request samples with these settings and the others
    # are greedy; the last one's disabled top-k sends its batch to the stock
    # path.
    for num_reqs, first in (
        (1, dict(top_k=64)),
        (8, dict(temperature=0.7, top_k=20, top_p=0.9)),
        (4, dict(temperature=1.3, top_k=50, min_p=0.05)),
        (2, dict(top_k=-1)),
    ):
        for q in range(num_reqs):
            kwargs = first if q == 0 else dict(temperature=0.0)
            params = SamplingParams(**kwargs, seed=1000 + q)
            state.add_request(q, SimpleNamespace(sampling_params=params))
            sampler.add_request(q, params)
        state.apply_staged_writes()
        sampler.apply_staged_writes()
        rows = num_reqs * (STEPS + 1)
        hidden = torch.randn(rows, hidden_size, device=device, generator=generator)
        hidden = hidden.bfloat16()
        zero_row = num_reqs == 1
        if zero_row:
            # Logits of exactly +0.0, whose key is 0x80000000: any key bit
            # pattern survives the exchange.
            hidden[0] = 0
            raw = model.compute_head_logits_local(hidden[:1])
            assert (raw.view(torch.int16) == 0).all()
        logits = model.compute_logits(hidden)
        noise = torch.randn(num_reqs, STEPS, vocab, device=device, generator=generator)
        draft.values[:num_reqs] = (
            logits.view(num_reqs, STEPS + 1, -1)[:, :STEPS] + noise / 2
        ).bfloat16()
        batch = _wiring_batch(num_reqs, generator, logits)

        assert not state.dummy_sampler_run(sampler, hidden)
        state.prepare_inputs(batch, req_states)
        sample_logits = state.compute_sample_logits(hidden)
        vocab_sharded = sample_logits.dtype == torch.int32
        assert vocab_sharded == (first["top_k"] != -1)
        got = rejection_sampler(sample_logits, batch, draft)
        if zero_row:
            # Every candidate of the row survives the exchange as +0.0 with
            # its column: the 64 smallest of rank 0's ties.
            lanes = rejection_sampler.verifier.kernels.lanes[0]
            assert torch.equal(lanes[0], torch.zeros_like(lanes[0]))
            assert lanes[1].tolist() == list(range(64))
        if vocab_sharded:
            outputs.append(got.sampled_token_ids.data_ptr())
        expected = stock(logits, batch, draft)
        # Greedy requests (and every request of a stock batch) match stock;
        # a sampled one accepts a prefix of its drafts.
        drafts = batch.input_ids.view(num_reqs, STEPS + 1)[:, 1:]
        for q in range(num_reqs):
            count = int(got.num_sampled[q])
            if q > 0 or not vocab_sharded:
                assert count == int(expected.num_sampled[q])
                assert torch.equal(
                    got.sampled_token_ids[q, :count],
                    expected.sampled_token_ids[q, :count],
                )
            assert 1 <= count <= STEPS + 1
            assert int(got.num_rejected[q]) == STEPS + 1 - count
            assert torch.equal(
                got.sampled_token_ids[q, : count - 1], drafts[q, : count - 1]
            )
    assert len(set(outputs)) == len(outputs) == 3, "outputs in flight reused"


@requires_sm100
@pytest.mark.distributed(num_gpus=2)
def test_model_state_wiring_matches_stock_verification():
    """Through `MuseSparkModelState`'s hooks at TP2, batches of greedy requests
    and requests sampling with a top-k of at most 64 verify vocab-sharded
    (greedy ones with the stock tokens and counts), each batch in flight in
    its own outputs, and a batch with top-k disabled takes the stock path. A
    row of +0.0 logits keeps all 64 candidates, at +0.0 with their columns,
    through the ranks' exchange."""
    run_tp2(_check_wiring)


def test_vocab_sharded_verifier_refuses_a_second_post_before_verify(monkeypatch):
    """A second post of candidate packets before the first one's verify
    (select) raises, also under `python -O`: it would pair the rank's later
    posts with the wrong selects."""
    from vllm.models.muse_spark.nvidia.specialized_decode.sample import (
        vocab_sharded,
    )

    verifier = object.__new__(vocab_sharded.VocabShardedVerifier)
    verifier._takes, verifier._posted = True, False
    verifier.model = SimpleNamespace(compute_head_logits_local=lambda h: h)
    monkeypatch.setattr(verifier, "_post", lambda raw: raw)
    hidden = torch.zeros(1, 1)
    verifier.logits(hidden)
    with pytest.raises(RuntimeError, match="never verified"):
        verifier.logits(hidden)


@requires_sm100
def test_vocab_sharded_select_keeps_a_candidate_of_a_nan_row():
    """A sampled row with a NaN logit keeps a candidate and writes in-vocab
    columns: its mask count must not reach 0, which placed its candidates at
    negative lanes, over the previous row's columns."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample import (
        verify as sharded,
    )

    rows = 2
    verify = _cached(
        sharded.ShardedVerify,
        SMALL_TP,
        SMALL_VOCAB,
        rows,
        SMALL_STEPS,
        sharded.STANDARD,
        "cuda",
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(rows, SMALL_VOCAB, device="cuda", generator=generator)
    logits[1, 5] = float("nan")
    states = torch.arange(rows, device="cuda")
    ones = torch.ones(rows, device="cuda")
    mailbox = _mailbox(SMALL_TP, rows)
    _post(_all_column_packets(logits), mailbox)
    verify.select(
        mailbox,
        rows,
        states,
        ones,
        torch.full((rows,), 64, dtype=torch.int32, device="cuda"),
        ones,
        torch.zeros(rows, device="cuda"),
    )
    lanes = verify.lanes[:rows].cpu()
    columns = lanes[:, 1]
    assert ((columns >= 0) & (columns < SMALL_VOCAB)).all()
    assert lanes[1, 0].view(torch.float32)[0] != float("-inf")


@requires_sm100
def test_vocab_sharded_select_waits_for_its_sequence_across_the_wrap():
    """Selects across the sequence numbers' wrap (126 to 1) wait for their
    packets, launched while the slots hold stale words (the tag of the call
    two before within the wrap, or its 7-bit `seq - 2`, which is 127 or 0 at
    the wrap), then read them and advance the numbers."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample import (
        verify as sharded,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.sample.shard_top_k import (  # noqa: E501
        SEQUENCES,
        TAG_MASK,
    )

    rows = 2
    verify = _cached(
        sharded.ShardedVerify,
        SMALL_TP,
        SMALL_VOCAB,
        rows,
        SMALL_STEPS,
        sharded.STANDARD,
        "cuda",
    )
    mailbox = _mailbox(SMALL_TP, rows)
    generator = torch.Generator(device="cuda").manual_seed(0)
    # Greedy rows (temperature 0) keep their 64 largest.
    zeros = torch.zeros(rows, device="cuda")
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    seqs = [SEQUENCES - 1, SEQUENCES]
    mailbox.seqs.copy_(torch.tensor(seqs))
    # Loaded before a select spins: a lazy module load waits for it.
    torch.cuda._sleep(1)
    for _ in range(3):
        stale = [(seqs[0] - 3) % SEQUENCES + 1, (seqs[1] - 2) & TAG_MASK]
        noise, logits = torch.randn(
            2, rows, SMALL_VOCAB, device="cuda", generator=generator
        )
        _post(_all_column_packets(noise), mailbox, torch.tensor(stale).cuda())
        side.wait_stream(main)
        verify.select(
            mailbox,
            rows,
            torch.arange(rows, device="cuda"),
            zeros,
            torch.full((rows,), 64, dtype=torch.int32, device="cuda"),
            zeros,
            zeros,
        )
        # The packets land after the select started polling.
        with torch.cuda.stream(side):
            torch.cuda._sleep(1 << 24)
            _post(_all_column_packets(logits), mailbox, torch.tensor(seqs).cuda())
        main.wait_stream(side)
        seqs = [s % SEQUENCES + 1 for s in seqs]
        assert mailbox.seqs.tolist() == seqs
        for row, lane in enumerate(verify.lanes[:rows]):
            columns = lane[1].long()
            top = logits[row].topk(64).indices.sort().values
            assert torch.equal(columns, top), (seqs, row)
            assert torch.equal(lane[0].view(torch.float32), logits[row, columns])
